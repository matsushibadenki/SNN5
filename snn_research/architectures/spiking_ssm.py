# ファイルパス: snn_research/architectures/spiking_ssm.py
# (新規作成)
#
# Title: Spiking State Space Model (SpikingSSM)
#
# Description:
# SNN5改善レポート (セクション5.3, 引用[8, 59]) に基づき、
# 長期時系列モデリング (Long-Range Arena SOTA) のための
# Spiking State Space Model (SpikingSSM) を実装します。
#
# これは Spiking Transformer よりも効率的に長期依存性を捉えることが
# 期待される次世代アーキテクチャです。
#
# mypy --strict 準拠。
#
# 改善 (SNN5改善レポート 5.3 対応):
# - S4DLIFBlock のスタブ実装を、SSMのRNNモード (h_t = A*h_{t-1} + B*x_t) と
#   LIFダイナミクスを組み合わせた、より忠実な実装に改善。
# - SpikingSSM.forward が T_seq (シーケンス長) で正しくループするように修正。
#
# 修正 (v2):
# - mypy [syntax] error: Unmatched '}' を解消。 (327行目)
#
# 修正 (v3):
# - mypy [name-defined] (Union) エラーを解消するため、Unionをインポート。
# - mypy [no-redef] (layer) エラーを解消するため、変数名を layer_to_reset に変更。
#
# 修正 (v4): SyntaxError: 末尾の余分な '}' を削除。
#
# 修正 (v_syn): SyntaxError: 末尾の不要な '}' を削除。
#
# 修正 (v_hpo_fix_attribute_error):
# - AttributeError: 'super' object has no attribute 'set_stateful' を修正。
# - super().set_stateful(stateful) を self.stateful = stateful に変更。

import torch
import torch.nn as nn
import torch.nn.functional as F
# --- ▼ 修正: Union をインポート ▼ ---
from typing import Tuple, Dict, Any, Type, Optional, List, cast, Union
# --- ▲ 修正 ▲ ---
import math

# SNNのコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
# --- ▼ 修正: SNN5改善レポートで追加したニューロンをインポート ▼ ---
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)
# --- ▲ 修正 ▲ ---
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]
from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]

import logging
logger = logging.getLogger(__name__)

class S4DLIFBlock(sj_base.MemoryModule):
    """
    (改善) S4D (Structured State Space) の計算を
    LIFニューロンのダイナミクスで近似するコアブロック。
    
    引用[8] (SpikingSSMs) に基づく。
    
    SSM RNNモード:
    h_t = A * h_{t-1} + B * x_t
    y_t = C * h_t + D * x_t
    
    SNN近似:
    A, B, C, D を学習可能な射影とし、
    状態 h_t と 出力 y_t をLIFニューロンでスパイク化する。
    """
    # --- ▼ 修正: 型ヒントを拡張 ▼ ---
    lif_h: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron] # 状態 h のニューロン
    lif_y: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron] # 出力 y のニューロン
    # --- ▲ 修正 ▲ ---

    def __init__(
        self,
        d_model: int, # D (入力/出力次元)
        d_state: int, # N (状態次元)
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any],
        d_conv: int = 4, # 畳み込み
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        # 1D畳み込み (x_t の前処理)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model, # Depthwise
            bias=True
        )
        
        # SSMパラメータの射影層
        # x_t -> B (入力行列用)
        self.in_proj_B = nn.Linear(d_model, d_state)
        # x_t -> D (ダイレクトパス用)
        self.in_proj_D = nn.Linear(d_model, d_model)
        # h_{t-1} -> A (状態遷移行列用)
        self.state_proj_A = nn.Linear(d_state, d_state)
        # h_t -> C (出力行列用)
        self.out_proj_C = nn.Linear(d_state, d_model)

        # SSMの核となるLIFニューロン (状態 h を管理)
        neuron_params_state = neuron_params.copy()
        neuron_params_state.pop('features', None)
        self.lif_h = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron], neuron_class(features=d_state, **neuron_params_state))
        
        # 出力用ニューロン (オプションだが、SpikingSSMのため追加)
        neuron_params_out = neuron_params.copy()
        neuron_params_out.pop('features', None)
        self.lif_y = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron], neuron_class(features=d_model, **neuron_params_out))

        self.norm = SNNLayerNorm(d_model)

    def set_stateful(self, stateful: bool) -> None:
        # --- ▼ 修正 (v_hpo_fix_attribute_error) ▼ ---
        # super().set_stateful(stateful) # 誤り
        self.stateful = stateful # 正しい
        # --- ▲ 修正 (v_hpo_fix_attribute_error) ▲ ---
        if hasattr(self.lif_h, 'set_stateful'):
            cast(Any, self.lif_h).set_stateful(stateful)
        if hasattr(self.lif_y, 'set_stateful'):
            cast(Any, self.lif_y).set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        if hasattr(self.lif_h, 'reset'):
            cast(Any, self.lif_h).reset()
        if hasattr(self.lif_y, 'reset'):
            cast(Any, self.lif_y).reset()

    def forward(
        self, 
        x_t: torch.Tensor, # (B, D_model)
        h_t_prev: torch.Tensor # (B, D_state) (前ステップの *スパイク*)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SSMの1ステップ更新 (RNNモード)。
        h_t = LIF( A(h_{t-1}_spike) + B(x_t) )
        y_t = LIF( C(h_t_spike) + D(x_t) )
        """
        
        # --- 1. 状態 h_t の計算 ---
        
        # A * h_{t-1} (状態の遷移)
        h_transition_current: torch.Tensor = self.state_proj_A(h_t_prev) 
        
        # B * x_t (入力の反映)
        h_input_current: torch.Tensor = self.in_proj_B(x_t)
        
        # h_t = LIF( A*h_{t-1} + B*x_t )
        h_t_spike, h_t_mem = self.lif_h(h_transition_current + h_input_current) # (B, D_state)
        
        # --- 2. 出力 y_t の計算 ---
        
        # C * h_t
        y_state_current: torch.Tensor = self.out_proj_C(h_t_spike)
        
        # D * x_t (ダイレクトパス)
        y_input_current: torch.Tensor = self.in_proj_D(x_t)
        
        # y_t = LIF( C*h_t + D*x_t )
        y_t_spike, _ = self.lif_y(y_state_current + y_input_current) # (B, D_model)
        
        # 残差接続 (x_t + y_t)
        y_t_out: torch.Tensor = self.norm(x_t + y_t_spike)
        
        return y_t_out, h_t_spike # 次のレイヤーへの入力, 次のステップの状態

# --- ▼ 修正: 構文エラーの原因となった '}' を削除 ▼ ---
# } # <-- この行を削除
# --- ▲ 修正 ▲ ---

class SpikingSSM(BaseModel):
    """
    Spiking State Space Model (SpikingSSM) アーキテクチャ。
    引用[8, 59]に基づく。
    """
    embedding: nn.Embedding
    pos_encoder: nn.Parameter
    layers: nn.ModuleList
    final_norm: SNNLayerNorm
    output_projection: nn.Linear
    
    # 1D Convolutional Layer (入力の前処理)
    conv1d_input: nn.Conv1d

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_state: int = 64,
        num_layers: int = 6,
        time_steps: int = 16, # (注: SNNの内部ステップ数, SSMでは未使用)
        d_conv: int = 4, # S4DLIFBlockに渡す畳み込みカーネルサイズ
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.time_steps = time_steps # SNNCore互換性のためのダミー
        
        if neuron_config is None:
            neuron_config = {'type': 'lif', 'tau_mem': 10.0, 'base_threshold': 1.0}
            
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        # --- ▼ 修正: SNN5改善レポートで追加したニューロンをサポート ▼ ---
        neuron_class: Type[nn.Module]
        filtered_params: Dict[str, Any] = {}

        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['tau_mem', 'base_threshold']}
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['a', 'b', 'c', 'd']}
        elif neuron_type_str == 'glif':
            neuron_class = GLIFNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['base_threshold']}
        elif neuron_type_str == 'tc_lif':
            neuron_class = TC_LIF
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']}
        elif neuron_type_str == 'dual_threshold':
            neuron_class = DualThresholdNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']}
        else:
            raise ValueError(f"Unknown neuron type for SpikingSSM: {neuron_type_str}")
        # --- ▲ 修正 ▲ ---

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1024, d_model)) # max_seq_len
        
        # 入力畳み込み (Mambaと同様)
        self.conv1d_input = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
            bias=True
        )

        self.layers = nn.ModuleList([
            S4DLIFBlock(d_model, d_state, neuron_class, filtered_params, d_conv=d_conv)
            for _ in range(num_layers)
        ])
        
        self.final_norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()
        logger.info(f"✅ SpikingSSM (S4D-LIF RNN Mode) initialized. (Layers: {num_layers}, D_State: {d_state})")

    def forward(
        self, 
        input_ids: torch.Tensor, # (B, T_seq)
        return_spikes: bool = False, 
        output_hidden_states: bool = False, # output_hidden_states を追加
        return_full_hiddens: bool = False, # return_full_hiddens を追加
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, T_seq = input_ids.shape
        device: torch.device = input_ids.device
        
        # 状態リセット
        SJ_F.reset_net(self)
        
        # 1. Analog Embedding
        x: torch.Tensor = self.embedding(input_ids) # (B, T_seq, D_model)
        x = x + self.pos_encoder[:, :T_seq, :]
        
        # 2. Input Convolution
        x_conv: torch.Tensor = self.conv1d_input(x.transpose(1, 2)) # (B, D_model, T_seq + K - 1)
        x_conv = x_conv[..., :T_seq].transpose(1, 2) # (B, T_seq, D_model)
        
        # (SpikingJellyの流儀に従い、LIFは外部でT_seqループを回す)
        
        # --- ▼ 修正: mypy [no-redef] (変数名を layer_to_set に変更) ▼ ---
        # 各レイヤーのLIFをStatefulに設定
        for layer_module in self.layers:
            layer_to_set: S4DLIFBlock = cast(S4DLIFBlock, layer_module)
            layer_to_set.set_stateful(True)
        # --- ▲ 修正 ▲ ---

        outputs: List[torch.Tensor] = []
        
        # 各レイヤーの状態 h を初期化 (スパイクなのでゼロ)
        h_states: List[torch.Tensor] = [
            torch.zeros(B, self.d_state, device=device) for _ in range(self.num_layers)
        ]
        
        # --- シーケンス長 (T_seq) でループ (RNNモード) ---
        for t_idx in range(T_seq):
            x_t: torch.Tensor = x_conv[:, t_idx, :] # (B, D_model)
            
            # --- レイヤー (num_layers) でループ ---
            x_t_layer: torch.Tensor = x_t
            
            for i in range(self.num_layers):
                layer: S4DLIFBlock = cast(S4DLIFBlock, self.layers[i])
                
                # h_t = A*h_{t-1} + B*x_t
                # y_t = C*h_t + D*x_t
                y_t, h_t_new = layer(x_t_layer, h_states[i])
                
                x_t_layer = y_t # 次のレイヤーへの入力 (残差接続はブロック内部)
                h_states[i] = h_t_new # 状態を更新
            
            outputs.append(x_t_layer) # 最終層の出力
        
        # --- ▼ 修正: mypy [no-redef] (変数名を layer_to_reset に変更) ▼ ---
        # 各レイヤーのLIFをStatelessに戻す
        for layer_module in self.layers:
            layer_to_reset: S4DLIFBlock = cast(S4DLIFBlock, layer_module)
            layer_to_reset.set_stateful(False)
        # --- ▲ 修正 ▲ ---

        # (B, T_seq, D_model)
        x_final_seq: torch.Tensor = torch.stack(outputs, dim=1)
        
        x_norm_final: torch.Tensor = self.final_norm(x_final_seq)
        
        output: torch.Tensor
        if output_hidden_states or return_full_hiddens:
            # (return_full_hiddens は T_snn 次元を期待するが、SSMは T_seq が時間軸)
            # (B, T_seq, D_model) を返す
            output = x_norm_final
        else:
            logits: torch.Tensor = self.output_projection(x_norm_final)
            output = logits
        
        # --- 互換性のため (logits, avg_spikes, mem) を返す ---
        # (T_snn ではなく T_seq で割る)
        avg_spikes_val: float = self.get_total_spikes() / (B * T_seq) if return_spikes and T_seq > 0 else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        mem: torch.Tensor = torch.tensor(0.0, device=device) 

        return output, avg_spikes, mem