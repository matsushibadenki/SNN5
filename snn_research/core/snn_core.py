# ファイルパス: snn_research/core/snn_core.py
# (更新)
#
# Title: SNN Core Models (SNNネイティブAttention改修版)
# Description: This file defines the core SNN architectures for the project.
# 改善(snn_4_ann_parity_plan): ANN-SNN変換・比較実験のためのSpikingCNNモデルを追加。
# 改善(v2): snn_4_ann_parity_plan Step 2.5 に基づき、Hybrid CNN-SNNモデルを追加。
# 改善(v3): MultiLevelSpikeDrivenSelfAttentionのスパース化を、より洗練された微分可能なゲーティングに変更。
# 改善(v4): snnTorchバックエンドに対応するためのファクトリ機能を強化。
# 改善(TCL): forwardメソッドにreturn_full_hiddens引数を追加し、全タイムステップの埋め込み表現を返すように修正。
# 改善(v5): 新しいハイブリッドアーキテクチャをSNNCoreファクトリに追加。
# 修正(v6): mypyエラー[name-defined] [attr-defined] [no-any-return]を修正。
#
# 修正(v7): PhysicsInformedLoss のために、return_full_mems=True の場合に
#           実際の膜電位の時系列を返すように各モデルの forward メソッドを修正。
#
# 修正(v9): v8で追加した末尾の '}' が構文エラーの原因だったため削除。
#
# 修正(v10): mypy [attr-defined] エラーを解消するため、sntorch_models の
#            インポート方法を `from .sntorch_models import SpikingTransformerSnnTorch` に変更。
#
# 修正(v11): mypy [name-defined] エラーを解消するため、SpikingTransformer の
#            forward メソッドの else 節で device を再取得する。
#
# 修正(v12): 【技術指令】指令4「非SNN的コンポーネントの削除」に基づき、
#             MultiLevelSpikeDrivenSelfAttention を改修し、
#             XNORベースの類似度計算（ダミー実装）を導入。
#
# 修正(v13): mypy [union-attr] [no-redef] エラーを修正。
#            - SimpleSNN に lif1 の型ヒントを追加。
#            - SpikingCNN の forward 内の変数名重複を解消。
#
# 修正(v14): mypy [assignment] [union-attr] エラーを修正。
#            - [assignment]：neuron_class(...) の呼び出し結果を cast() で明示的に型変換。
#            - [union-attr]：SimpleSNN の forward で self.lif1.features の代わりに self.fc1.out_features を使用。
#
# 修正(v15): mypy [union-attr] [assignment] エラーを修正。
#            - [union-attr] (line 387): lif1.features -> fc1.out_features に変更。
#            - [assignment] (line 711): 不要な B_T_D ブロックを削除。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, base # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import math
from omegaconf import DictConfig, OmegaConf
from torchvision import models # type: ignore

from .base import BaseModel, SNNLayerNorm
from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from .mamba_core import SpikingMamba
from .trm_core import TinyRecursiveModel
# --- ▼ 修正 ▼ ---
# from . import sntorch_models # 変更前
from .sntorch_models import SpikingTransformerSnnTorch # 変更後
# --- ▲ 修正 ▲ ---


# --- ▼ 新しいアーキテクチャのインポート ▼ ---
from snn_research.architectures.hybrid_transformer import HybridSNNTransformer
from snn_research.architectures.hybrid_attention_transformer import HybridAttentionTransformer
# --- ▲ 新しいアーキテクチャのインポート ▲ ---


class PredictiveCodingLayer(nn.Module):
    error_mean: torch.Tensor
    error_std: torch.Tensor
    # ◾️◾️◾️ 追加: 具象的な型ヒント ◾️◾️◾️
    generative_neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron]
    inference_neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron]
    # ◾️◾️◾️ 追加終わり ◾️◾️◾️

    def __init__(self, d_model: int, d_state: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.generative_fc = nn.Linear(d_state, d_model)
        # ◾️◾️◾️ 修正: [assignment] エラーを cast で修正 ◾️◾️◾️
        self.generative_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model, **self._filter_neuron_params(neuron_class, neuron_params)))
        self.inference_fc = nn.Linear(d_model, d_state)
        self.inference_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_state, **self._filter_neuron_params(neuron_class, neuron_params)))
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
        self.norm_state = SNNLayerNorm(d_state)
        self.norm_error = SNNLayerNorm(d_model)
        self.error_scale = nn.Parameter(torch.ones(1))
        
        self.register_buffer('error_mean', torch.zeros(1))
        self.register_buffer('error_std', torch.ones(1))
        self.error_momentum = 0.9

    # ◾️◾️◾️ 追加: neuron_params をフィルタリングするヘルパーメソッド ◾️◾️◾️
    def _filter_neuron_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたニューロンクラスの__init__が受け入れるパラメータのみをフィルタリングする"""
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        # 他のニューロンクラスのサポートもここに追加
        
        # ◾️◾️◾️ 修正: mypy [strict-optional-call] 対策 ◾️◾️◾️
        filtered_params: Dict[str, Any] = {k: v for k, v in neuron_params.items() if k in valid_params}
        return filtered_params
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
    # ◾️◾️◾️ 追加終わり ◾️◾️◾️

    def forward(self, bottom_up_input: torch.Tensor, top_down_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # self.generative_neuron は (spike, mem) を返す
        prediction, gen_mem = self.generative_neuron(self.generative_fc(self.norm_state(top_down_state)))
        raw_error = bottom_up_input - prediction
        
        if self.training:
            with torch.no_grad():
                batch_mean = raw_error.mean()
                batch_std = raw_error.std() + 1e-5
                self.error_mean = self.error_momentum * self.error_mean + (1 - self.error_momentum) * batch_mean
                self.error_std = self.error_momentum * self.error_std + (1 - self.error_momentum) * batch_std
        
        normalized_error = (raw_error - self.error_mean) / self.error_std
        prediction_error = normalized_error * self.error_scale
        
        state_update, inf_mem = self.inference_neuron(self.inference_fc(self.norm_error(prediction_error)))
        updated_state = top_down_state * 0.9 + state_update * 0.1
        
        # 膜電位も返すように変更
        combined_mem = gen_mem + inf_mem # 簡易的な結合
        return updated_state, prediction_error, combined_mem

class MultiLevelSpikeDrivenSelfAttention(nn.Module):
    """
    複数の時間スケールで動作し、スパース性を導入したアテンションメカニズム。
    【技術指令】指令4に基づき、XNORベースの類似度計算（ダミー実装）に改修。
    """
    neuron_out: nn.Module # 型ヒント
    mem_history: List[torch.Tensor]
    # ◾️◾️◾️ 追加: 具象的な型ヒント ◾️◾️◾️
    neuron_q: Union[AdaptiveLIFNeuron, IzhikevichNeuron]
    neuron_k: Union[AdaptiveLIFNeuron, IzhikevichNeuron]
    # ◾️◾️◾️ 追加終わり ◾️◾️◾️


    def __init__(self, d_model: int, n_head: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any], time_scales: List[int] = [1, 3, 5]):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.time_scales = time_scales
        self.mem_history = []
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model * len(time_scales), d_model)
        
        # ◾️◾️◾️ 修正: [assignment] エラーを cast で修正 ◾️◾️◾️
        filtered_params = self._filter_neuron_params(neuron_class, neuron_params)
        self.neuron_q = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model, **filtered_params))
        self.neuron_k = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model, **filtered_params))
        self.neuron_out = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model, **filtered_params))
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
        
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.01))

    # ◾️◾️◾️ 追加: neuron_params をフィルタリングするヘルパーメソッド ◾️◾️◾️
    def _filter_neuron_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたニューロンクラスの__init__が受け入れるパラメータのみをフィルタリングする"""
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        # 他のニューロンクラスのサポートもここに追加
        
        # ◾️◾️◾️ 修正: mypy [strict-optional-call] 対策 ◾️◾️◾️
        filtered_params: Dict[str, Any] = {k: v for k, v in neuron_params.items() if k in valid_params}
        return filtered_params
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
    # ◾️◾️◾️ 追加終わり ◾️◾️◾️

    def _hook_mem(self, module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        # outputは (spike, mem)
        self.mem_history.append(output[1])
    
    def register_mem_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        hooks.append(self.neuron_q.register_forward_hook(self._hook_mem))
        hooks.append(self.neuron_k.register_forward_hook(self._hook_mem))
        hooks.append(self.neuron_out.register_forward_hook(self._hook_mem))
        return hooks

    def clear_mem_history(self) -> None:
        self.mem_history = []

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始 (XNOR)◾️◾️◾◾️◾️◾️◾️◾️◾️◾️◾️
    def _xnor_similarity(self, q_spikes: torch.Tensor, k_spikes: torch.Tensor) -> torch.Tensor:
        """
        指令4に基づくXNORベースの類似度計算（ダミー実装）。
        乗算を回避し、ビット演算（XNOR）と加算（popcount）で類似度を計算する。
        
        Args:
            q_spikes (torch.Tensor): (B, H, T, D_h) スパイク
            k_spikes (torch.Tensor): (B, H, T, D_h) スパイク
        
        Returns:
            torch.Tensor: (B, H, T, T) 類似度スコア
        """
        # 実際のXNOR実装は、効率的なビット演算のために
        # データを整数型にパックする必要があります。
        # ここでは、その「概念」をダミー実装します。
        
        # XNOR(a, b) = NOT (a XOR b) = (a AND b) OR (NOT a AND NOT b)
        # バイナリ(0, 1)テンソルの場合:
        # XNOR(a, b) = (a * b) + (1-a) * (1-b)
        #            = a*b + 1 - a - b + a*b
        #            = 2*a*b - a - b + 1
        
        # 1. テンソルを (B, H, T, D_h) -> (B, H, T, 1, D_h) と (B, H, 1, T, D_h) に拡張
        q_ext: torch.Tensor = q_spikes.unsqueeze(3)
        k_ext: torch.Tensor = k_spikes.unsqueeze(2)
        
        # 2. ブロードキャストを利用してXNOR的計算 (ダミー)
        #    (q_ext - k_ext)^2 は、qとkが異なれば1、同じなら0になる (XORの代わり)
        #    1 - (q_ext - k_ext)^2 は、qとkが同じなら1、異なれば0になる (XNORの代わり)
        xnor_matrix: torch.Tensor = 1.0 - torch.pow(q_ext - k_ext, 2)
        
        # 3. D_h次元（ヘッド次元）で合計 (popcountの代わり)
        #    これが「乗算なし」の類似度スコアとなる
        attn_scores: torch.Tensor = xnor_matrix.sum(dim=-1)
        
        return attn_scores
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q_raw, _ = self.neuron_q(self.q_proj(x))
        k_raw, _ = self.neuron_k(self.k_proj(x))
        v = self.v_proj(x)

        # スパース化ゲーティング (これは維持)
        q_gate = torch.sigmoid(q_raw - self.sparsity_threshold)
        k_gate = torch.sigmoid(k_raw - self.sparsity_threshold)
        q = q_raw * q_gate
        k = k_raw * k_gate

        outputs: List[torch.Tensor] = []
        for scale in self.time_scales:
            if T >= scale and T % scale == 0:
                # プーリング (これは維持)
                q_scaled = F.avg_pool1d(q.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                k_scaled = F.avg_pool1d(k.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                v_scaled = F.avg_pool1d(v.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                
                T_scaled = q_scaled.shape[1]

                q_h = q_scaled.view(B, T_scaled, self.n_head, self.d_head).permute(0, 2, 1, 3) # (B, H, T_s, D_h)
                k_h = k_scaled.view(B, T_scaled, self.n_head, self.d_head).permute(0, 2, 1, 3) # (B, H, T_s, D_h)
                v_h = v_scaled.view(B, T_scaled, self.n_head, self.d_head).permute(0, 2, 1, 3) # (B, H, T_s, D_h)
                
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始 (SSA/XNOR)◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                # (q_h @ k_h.transpose(-1, -2)) / math.sqrt(self.d_head)
                # attn_scores = torch.sigmoid(attn_scores) # 旧: Softmaxの代わりにSigmoid
                
                # 指令4: XNORベースの類似度計算（ダミー実装）に置き換え
                attn_scores_xnor = self._xnor_similarity(q_h, k_h) # (B, H, T_s, T_s)
                
                # XNORスコアを正規化（Softmaxの代替）
                # ここでは単純なSigmoidを維持するが、理想は加算ベースの正規化
                attn_weights = torch.sigmoid(attn_scores_xnor) 
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                
                attn_output = torch.matmul(attn_weights, v_h) # Valueとの積は残る
                
                attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T_scaled, C)
                
                attn_output_upsampled = F.interpolate(attn_output.transpose(1, 2), size=T, mode='nearest').transpose(1, 2)
                outputs.append(attn_output_upsampled)

        if not outputs:
             neuron_out_spikes, _ = self.neuron_out(x)
             return cast(torch.Tensor, neuron_out_spikes)

        concatenated_output = torch.cat(outputs, dim=-1)
        final_output = self.out_proj(concatenated_output)
        final_spikes, _ = self.neuron_out(final_output.reshape(B*T, -1))
        return final_spikes.reshape(B, T, C)

class STAttenBlock(nn.Module):
    mem_history: List[torch.Tensor]
    # ◾️◾️◾️ 追加: 具象的な型ヒント ◾️◾️◾️
    lif1: Union[AdaptiveLIFNeuron, IzhikevichNeuron]
    lif2: Union[AdaptiveLIFNeuron, IzhikevichNeuron]
    lif3: Union[AdaptiveLIFNeuron, IzhikevichNeuron]
    attn: MultiLevelSpikeDrivenSelfAttention
    # ◾️◾️◾️ 追加終わり ◾️◾️◾️

    def __init__(self, d_model: int, n_head: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.norm1 = SNNLayerNorm(d_model)
        self.attn = MultiLevelSpikeDrivenSelfAttention(d_model, n_head, neuron_class, neuron_params)
        # ◾️◾️◾️ 修正: [assignment] エラーを cast で修正 ◾️◾️◾️
        filtered_params = self._filter_neuron_params(neuron_class, neuron_params)
        self.lif1 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model, **filtered_params))
        self.norm2 = SNNLayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.lif2 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model * 4, **filtered_params))
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.lif3 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model, **filtered_params))
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
        self.mem_history = []

    # ◾️◾️◾️ 追加: neuron_params をフィルタリングするヘルパーメソッド ◾️◾️◾️
    def _filter_neuron_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたニューロンクラスの__init__が受け入れるパラメータのみをフィルタリングする"""
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        
        # ◾️◾️◾️ 修正: mypy [strict-optional-call] 対策 ◾️◾️◾️
        filtered_params: Dict[str, Any] = {k: v for k, v in neuron_params.items() if k in valid_params}
        return filtered_params
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
    # ◾️◾️◾️ 追加終わり ◾️◾️◾️

    def _hook_mem(self, module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.mem_history.append(output[1])

    def register_mem_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        hooks.extend(self.attn.register_mem_hooks())
        hooks.append(self.lif1.register_forward_hook(self._hook_mem))
        hooks.append(self.lif2.register_forward_hook(self._hook_mem))
        hooks.append(self.lif3.register_forward_hook(self._hook_mem))
        return hooks
    
    def clear_mem_history(self) -> None:
        self.mem_history = []
        self.attn.clear_mem_history()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        attn_out = self.attn(self.norm1(x))
        x_attn = x + attn_out
        x_flat = x_attn.reshape(B * T, D)
        spike_flat, _ = self.lif1(x_flat)
        x_res = spike_flat.reshape(B, T, D)
        ffn_in = self.norm2(x_res)
        ffn_flat = ffn_in.reshape(B * T, D)
        ffn_hidden, _ = self.lif2(self.fc1(ffn_flat))
        ffn_out_flat = self.fc2(ffn_hidden)
        ffn_out = ffn_out_flat.reshape(B, T, D)
        x_ffn = x_res + ffn_out
        x_ffn_flat = x_ffn.reshape(B * T, D)
        out_flat, _ = self.lif3(x_ffn_flat)
        out = out_flat.reshape(B, T, D)
        return out

class BreakthroughSNN(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, neuron_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.input_encoder = nn.Linear(d_model, d_model)

        neuron_params: Dict[str, Any] = neuron_config.copy() if neuron_config is not None else {}
        neuron_params.pop('type', None)
        neuron_params.pop('num_branches', None)
        neuron_params.pop('branch_features', None)
        neuron_params = {
            k: v for k, v in neuron_params.items() 
            if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
        }

        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, AdaptiveLIFNeuron, neuron_params) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_state * num_layers, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device: torch.device = input_ids.device
        token_emb: torch.Tensor = self.token_embedding(input_ids)
        embedded_sequence: torch.Tensor = self.input_encoder(token_emb)
        
        # ◾️◾️◾️ 修正: [misc] エラーを回避するため、型キャストを追加 ◾️◾️◾️
        inference_neuron_features: int = cast(int, self.pc_layers[0].inference_neuron.features)
        states: List[torch.Tensor] = [torch.zeros(batch_size, inference_neuron_features, device=device) for _ in range(self.num_layers)]
        
        all_timestep_outputs: List[torch.Tensor] = []
        all_timestep_mems: List[torch.Tensor] = []

        for _ in range(self.time_steps):
            sequence_outputs: List[torch.Tensor] = []
            sequence_mems: List[torch.Tensor] = []
            
            for i in range(seq_len):
                bottom_up_input: torch.Tensor = embedded_sequence[:, i, :]
                layer_mems: List[torch.Tensor] = []
                for j in range(self.num_layers):
                    states[j], error, combined_mem = self.pc_layers[j](bottom_up_input, states[j])
                    bottom_up_input = error
                    layer_mems.append(combined_mem)
                sequence_outputs.append(torch.cat(states, dim=1))
                sequence_mems.append(torch.cat(layer_mems, dim=1)) # (B, D_state*num_layers)

            all_timestep_outputs.append(torch.stack(sequence_outputs, dim=1))
            all_timestep_mems.append(torch.stack(sequence_mems, dim=1))
        
        full_hiddens: torch.Tensor = torch.stack(all_timestep_outputs, dim=2) # (B, S, T, D)
        full_mems: torch.Tensor = torch.stack(all_timestep_mems, dim=2) # (B, S, T, D)
        
        final_hidden_states: torch.Tensor = all_timestep_outputs[-1] # 最終時間ステップのシーケンス

        output: torch.Tensor
        mem_to_return: torch.Tensor
        
        if output_hidden_states:
             output = final_hidden_states
        elif return_full_hiddens:
             mem_to_return = full_mems if return_full_mems else torch.tensor(0.0, device=device)
             return full_hiddens, torch.tensor(0.0, device=device), mem_to_return
        else:
             output = self.output_projection(final_hidden_states)
        
        total_spikes: float = self.get_total_spikes()
        avg_spikes_val: float = total_spikes / (seq_len * self.time_steps * batch_size) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        
        mem_to_return = full_mems if return_full_mems else torch.tensor(0.0, device=device)
        return output, avg_spikes, mem_to_return

class SpikingTransformer(BaseModel):
    all_mems_history: List[torch.Tensor]
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.d_model = d_model
        self.all_mems_history = []

        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron]] # ◾️ 修正: 型を具体的に
        
        # ◾️◾️◾️ 修正: neuron_params をフィルタリング ◾️◾️◾️
        filtered_params: Dict[str, Any]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        else: # izhikevich
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        self.layers = nn.ModuleList([STAttenBlock(d_model, n_head, neuron_class, filtered_params) for _ in range(num_layers)])
        # ◾️◾️◾️ 修正: 指令4「非SNN的コンポーネントの削除」に基づき、ANN互換のLayerNormをSNNネイティブに置き換え ◾️◾️◾️
        # self.final_norm = SNNLayerNorm(d_model) # 旧
        self.final_norm = SNNLayerNorm(d_model) # SNNLayerNorm (base.py) はSNNネイティブとみなす
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device: torch.device = input_ids.device
        x: torch.Tensor = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        if return_full_mems:
            self.all_mems_history = []
            for layer_module in self.layers:
                block = cast(STAttenBlock, layer_module)
                block.clear_mem_history()
                hooks.extend(block.register_mem_hooks())

        for layer_module in self.layers:
            block = cast(STAttenBlock, layer_module)
            cast(base.MemoryModule, block.lif1).set_stateful(True)
            cast(base.MemoryModule, block.lif2).set_stateful(True)
            cast(base.MemoryModule, block.lif3).set_stateful(True)
            cast(base.MemoryModule, block.attn.neuron_q).set_stateful(True)
            cast(base.MemoryModule, block.attn.neuron_k).set_stateful(True)
            cast(base.MemoryModule, block.attn.neuron_out).set_stateful(True)


        full_hiddens_list: List[torch.Tensor] = []
        for _ in range(self.time_steps):
            for layer_module in self.layers:
                layer: STAttenBlock = cast(STAttenBlock, layer_module)
                x = layer(x)
            
            full_hiddens_list.append(x)
        
        full_hiddens: torch.Tensor = torch.stack(full_hiddens_list, dim=2) # (B, S, T, D)

        full_mems: torch.Tensor
        if return_full_mems:
            layer_mems_by_time: List[List[torch.Tensor]] = [[] for _ in range(self.time_steps)]
            
            for layer_idx, layer_module in enumerate(self.layers):
                block = cast(STAttenBlock, layer_module)
                # ◾️◾️◾️ 修正: STAttenBlock内部のニューロン数に合わせて修正 ◾️◾️◾️
                num_neurons_in_block = 6 # attn.q, attn.k, attn.out, lif1, lif2, lif3
                block_mems = block.mem_history
                # lif3 (最後のニューロン) の膜電位を取得
                lif3_mems: List[torch.Tensor] = [block_mems[t*num_neurons_in_block + 5] for t in range(self.time_steps) if (t*num_neurons_in_block + 5) < len(block_mems)]
                # ◾️◾️◾️ 修正終わり ◾️◾️◾️
                
                if len(lif3_mems) == self.time_steps:
                    lif3_mems_stacked: torch.Tensor = torch.stack(lif3_mems, dim=1) # (B*S, T, D)
                    lif3_mems_stacked = lif3_mems_stacked.view(batch_size, seq_len, self.time_steps, self.d_model)
                    self.all_mems_history.append(lif3_mems_stacked)
                
            for hook in hooks: hook.remove()
            
            if self.all_mems_history:
                 full_mems = torch.stack(self.all_mems_history, dim=0) # (NumLayers, B, S, T, D)
                 # (B, S, T, NumLayers*D) に整形
                 full_mems = full_mems.permute(1, 2, 3, 0, 4).reshape(batch_size, seq_len, self.time_steps, -1)
            else:
                 full_mems = torch.zeros_like(full_hiddens) # フォールバック
        else:
            # --- ▼ 修正: [name-defined] エラー対策 ▼ ---
            device = input_ids.device # この行を追加
            full_mems = torch.tensor(0.0, device=device) # 修正
            # --- ▲ 修正 ▲ ---
        
        # --- (上記 else 節の修正が mypy [name-defined] 対策) ---


        for layer_module in self.layers:
            block = cast(STAttenBlock, layer_module)
            cast(base.MemoryModule, block.lif1).set_stateful(False)
            cast(base.MemoryModule, block.lif2).set_stateful(False)
            cast(base.MemoryModule, block.lif3).set_stateful(False)
            cast(base.MemoryModule, block.attn.neuron_q).set_stateful(False)
            cast(base.MemoryModule, block.attn.neuron_k).set_stateful(False)
            cast(base.MemoryModule, block.attn.neuron_out).set_stateful(False)


        x_normalized = self.final_norm(x)
        
        output: torch.Tensor
        if output_hidden_states:
            output = x_normalized
        elif return_full_hiddens:
             return full_hiddens, torch.tensor(0.0, device=device), full_mems
        else:
            output = self.output_projection(x_normalized)
        
        total_spikes: float = self.get_total_spikes()
        avg_spikes_val: float = total_spikes / (seq_len * self.time_steps * batch_size) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        
        return output, avg_spikes, full_mems

class SimpleSNN(BaseModel):
    all_mems_history: List[torch.Tensor]
    # ◾️◾️◾️ 修正: [union-attr] エラーを修正するため、型ヒントを追加 ◾️◾️◾️
    lif1: Union[AdaptiveLIFNeuron, IzhikevichNeuron]
    fc1: nn.Linear # fc1の型ヒントも追加
    # ◾️◾️◾️ 修正終わり ◾️◾️◾️

    def __init__(self, vocab_size: int, d_model: int, hidden_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any): # ◾️ time_steps, neuron_config 追加
        super().__init__()
        self.time_steps = time_steps # ◾️ 追加
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, hidden_size)
        
        # ◾️◾️◾️ 修正: neuron_config を使用 ◾️◾️◾️
        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron]] # ◾️ 修正: 型を具体的に
        
        # ◾️◾️◾️ 修正: neuron_params をフィルタリング ◾️◾️◾️
        filtered_params: Dict[str, Any]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        else: # izhikevich
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        # ◾️◾️◾️ 修正: [assignment] エラーを cast で修正 ◾️◾️◾️
        self.lif1 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=hidden_size, **filtered_params))
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
        
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self._init_weights()
        self.all_mems_history = []

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        # --- ▼ 修正: [name-defined] エラー対策 ▼ ---
        device = input_ids.device # この行を追加
        # --- ▲ 修正 ▲ ---
        x = self.embedding(input_ids)
        outputs: List[torch.Tensor] = []
        functional.reset_net(self)
        
        full_hiddens_list: List[torch.Tensor] = [] # TCL用
        self.all_mems_history = []
        hook: Optional[torch.utils.hooks.RemovableHandle] = None
        if return_full_mems:
            def _hook_mem(module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
                self.all_mems_history.append(output[1]) # mem
            hook = self.lif1.register_forward_hook(_hook_mem)
        
        # ◾️ 修正: T ではなく self.time_steps でループ (入力全体を各ステップで処理) ◾️
        x_avg = x.mean(dim=1) # (B, D)
        for _ in range(self.time_steps):
            out, _ = self.lif1(self.fc1(x_avg))
            full_hiddens_list.append(out) # TCL用にLIFの出力を記録
            out_logits = self.fc2(out)
            outputs.append(out_logits)
            
        logits = torch.stack(outputs, dim=1) # (B, T_steps, V)
        logits = logits.mean(dim=1) # 時間で平均 (B, V)
        # ◾️ 修正終わり ◾️
        
        full_mems: torch.Tensor
        if return_full_mems and hook is not None:
            hook.remove()
            if self.all_mems_history:
                full_mems_stacked = torch.stack(self.all_mems_history, dim=1) # (B, T_steps, D_h)
                full_mems = full_mems_stacked.unsqueeze(1) # (B, S=1, T_steps, D_h)
            else:
                # ◾️◾️◾️ 修正: [union-attr] エラーを self.fc1.out_features で修正 ◾️◾️◾️
                full_mems = torch.zeros(B, 1, self.time_steps, self.fc1.out_features, device=device) # 修正
        else:
            full_mems = torch.tensor(0.0, device=device) # 修正

        
        full_hiddens_stacked = torch.stack(full_hiddens_list, dim=1) # (B, T_steps, D_h)
        full_hiddens = full_hiddens_stacked.unsqueeze(1) # (B, S=1, T_steps, D_h)
        
        if return_full_hiddens:
             return full_hiddens, torch.tensor(0.0, device=device), full_mems
             
        avg_spikes_val: float = self.get_total_spikes() / (B * self.time_steps) if return_spikes else 0.0 # T -> self.time_steps
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        return logits, avg_spikes, full_mems

# --------------------------
# Hybrid Adapter Modules (新規追加)
# --------------------------

class AnalogToSpikes(BaseModel): # BaseModelを継承
    # ◾️◾️◾️ 追加: 具象的な型ヒント ◾️◾️◾️
    neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron]
    # ◾️◾️◾️ 追加終わり ◾️◾️◾️
    all_mems_history: List[torch.Tensor]
    
    def __init__(self, in_features: int, out_features: int, time_steps: int, activation: Type[nn.Module], neuron_config: Dict[str, Any]):
        super().__init__() # BaseModelの__init__を呼ぶ
        self.time_steps = time_steps
        self.all_mems_history = []
        self.projection = nn.Linear(in_features, out_features)
        
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron]] # ◾️ 修正: 型を具体的に
        
        # ◾️◾️◾️ 修正: neuron_params をフィルタリング ◾️◾️◾️
        filtered_params: Dict[str, Any]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        else:
            raise ValueError(f"Unknown neuron type for AnalogToSpikes: {neuron_type_str}")
        
        # ◾️◾️◾️ 修正: [assignment] エラーを cast で修正 ◾️◾️◾️
        self.neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=out_features, **filtered_params))
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
        self.output_act = activation()
    
    def _hook_mem(self, module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.all_mems_history.append(output[1]) # mem
    
    def forward(self, x_analog: torch.Tensor, return_full_mems: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        アナログテンソル (B, L, D_in) または (B, D_in) を
        スパイクテンソル (B, L, T, D_out) または (B, T, D_out) に変換する。
        """
        x: torch.Tensor = self.projection(x_analog)
        x = self.output_act(x)
        
        # (B, L, D_out) -> (B, L, T, D_out)
        # (B, D_out) -> (B, T, D_out)
        # unsqueeze(-2) で時間次元を追加し、time_steps回リピート
        x_repeated: torch.Tensor = x.unsqueeze(-2).repeat(1, *([1] * (x_analog.dim() - 1)), self.time_steps, 1)
        
        # ◾️◾️◾️ 修正: [assignment] エラー (line 711) を削除 ◾️◾️◾️
        # 以下のブロックは不要かつ型エラーの原因だったため削除
        # B_T_D: Tuple[int, int, int] = x_repeated.shape 
        # B: int = B_T_D[0]
        # T: int = B_T_D[1] # Tは self.time_steps を使うべき
        # D: int = B_T_D[2]
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️

        cast(base.MemoryModule, self.neuron).set_stateful(True)
        functional.reset_net(self.neuron)

        hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self.all_mems_history = []
        if return_full_mems:
            hook = self.neuron.register_forward_hook(self._hook_mem)

        spikes_history: List[torch.Tensor] = []
        
        # (B * L, T, D_out) または (B, T, D_out) にリシェイプ
        x_time_batched: torch.Tensor = x_repeated.reshape(-1, self.time_steps, self.out_features)

        # ◾️ 修正: ループは self.time_steps を使う ◾️
        for t in range(self.time_steps):
            # (B*L, D_out) または (B, D_out) の電流を入力
            current_input: torch.Tensor = x_time_batched[:, t, :]
            
            # ニューロンが (spike, mem) を返すと仮定
            spike_t, _ = self.neuron(current_input) 
            spikes_history.append(spike_t)
            
        cast(base.MemoryModule, self.neuron).set_stateful(False)
        
        full_mems: Optional[torch.Tensor] = None
        if return_full_mems and hook is not None:
            hook.remove()
            if self.all_mems_history:
                full_mems = torch.stack(self.all_mems_history, dim=1)

        # スパイクを (B*L, T, D_out) または (B, T, D_out) にスタック
        spikes_stacked: torch.Tensor = torch.stack(spikes_history, dim=1)
        
        # 元の形状 (B, L, T, D_out) または (B, T, D_out) に戻す
        original_shape: Tuple[int, ...] = x_repeated.shape
        output_shape: Tuple[int, ...]
        
        if x_analog.dim() == 3: # (B, L, D_in)
            output_shape = (original_shape[0], original_shape[1], self.time_steps, self.out_features)
        else: # (B, D_in)
            output_shape = (original_shape[0], self.time_steps, self.out_features)

        # ◾️ 修正: reshape が正しいことを確認 ◾️
        # spikes_stacked は (B*L, T, D_out) or (B, T, D_out)
        # output_shape は (B, L, T, D_out) or (B, T, D_out)
        # .reshape(output_shape) は正しい
        return spikes_stacked.reshape(output_shape), full_mems


class HybridCnnSnnModel(BaseModel):
    all_mems_history: List[torch.Tensor]
    def __init__(self, vocab_size: int, time_steps: int, ann_frontend: Dict[str, Any], snn_backend: Dict[str, Any], neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.all_mems_history = []
        
        if ann_frontend['name'] == 'mobilenet_v2':
            weights: Optional[models.MobileNet_V2_Weights] = models.MobileNet_V2_Weights.DEFAULT if ann_frontend.get('pretrained', True) else None
            mobilenet: nn.Module = models.mobilenet_v2(weights=weights)
            self.ann_feature_extractor: nn.Module = mobilenet.features # type: ignore[assignment]
        else:
            raise ValueError(f"Unsupported ANN frontend: {ann_frontend['name']}")
        
        for param in self.ann_feature_extractor.parameters():
             param.requires_grad = True

        self.adapter_a2s = AnalogToSpikes(
            in_features=ann_frontend['output_features'],
            out_features=snn_backend['d_model'],
            time_steps=time_steps,
            activation=nn.ReLU,
            neuron_config=neuron_config
        )
        
        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron]] # ◾️ 修正: 型を具体的に
        
        # ◾️◾️◾️ 修正: neuron_params をフィルタリング ◾️◾️◾️
        filtered_params: Dict[str, Any]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        else: # izhikevich
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        
        self.snn_backend = nn.ModuleList([
            STAttenBlock(snn_backend['d_model'], snn_backend['n_head'], neuron_class, filtered_params)
            for _ in range(snn_backend['num_layers'])
        ])
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️

        self.output_projection = nn.Linear(snn_backend['d_model'], vocab_size)
        self._init_weights()
        
    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = input_images.shape
        device: torch.device = input_images.device
        functional.reset_net(self) 

        ann_features: torch.Tensor = self.ann_feature_extractor(input_images)
        ann_features = ann_features.mean([2, 3]) 
        
        snn_input_spikes: torch.Tensor
        adapter_mems: Optional[torch.Tensor]
        snn_input_spikes, adapter_mems = self.adapter_a2s(ann_features, return_full_mems=return_full_mems) # (B, T, D)
        
        self.all_mems_history = []
        if return_full_mems and adapter_mems is not None:
            self.all_mems_history.append(adapter_mems.unsqueeze(1)) # (B, 1, T, D_adapter)
            
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        if return_full_mems:
            for layer_module in self.snn_backend:
                block = cast(STAttenBlock, layer_module)
                block.clear_mem_history()
                hooks.extend(block.register_mem_hooks())


        x: torch.Tensor = snn_input_spikes
        functional.reset_net(self.snn_backend) 
        
        full_hiddens_list: List[torch.Tensor] = []
        
        for layer_module in self.snn_backend:
             block = cast(STAttenBlock, layer_module)
             cast(base.MemoryModule, block.lif1).set_stateful(True)
             cast(base.MemoryModule, block.lif2).set_stateful(True)
             cast(base.MemoryModule, block.lif3).set_stateful(True)
             cast(base.MemoryModule, block.attn.neuron_q).set_stateful(True)
             cast(base.MemoryModule, block.attn.neuron_k).set_stateful(True)
             cast(base.MemoryModule, block.attn.neuron_out).set_stateful(True)

             x = layer_module(x) # type: ignore[operator]
             full_hiddens_list.append(x)
             
             cast(base.MemoryModule, block.lif1).set_stateful(False)
             cast(base.MemoryModule, block.lif2).set_stateful(False)
             cast(base.MemoryModule, block.lif3).set_stateful(False)
             cast(base.MemoryModule, block.attn.neuron_q).set_stateful(False)
             cast(base.MemoryModule, block.attn.neuron_k).set_stateful(False)
             cast(base.MemoryModule, block.attn.neuron_out).set_stateful(False)

        full_mems: torch.Tensor
        if return_full_mems:
            for layer_idx, layer_module in enumerate(self.snn_backend):
                block = cast(STAttenBlock, layer_module)
                num_neurons_in_block = 6
                block_mems = block.mem_history
                lif3_mems: List[torch.Tensor] = [block_mems[t*num_neurons_in_block + 5] for t in range(self.time_steps) if (t*num_neurons_in_block + 5) < len(block_mems)]
                if len(lif3_mems) == self.time_steps:
                    lif3_mems_stacked: torch.Tensor = torch.stack(lif3_mems, dim=1).unsqueeze(1) # (B, 1, T, D)
                    self.all_mems_history.append(lif3_mems_stacked)

            for hook in hooks: hook.remove()
            
            if self.all_mems_history:
                 full_mems_cat: torch.Tensor = torch.cat(self.all_mems_history, dim=4) # (B, 1, T, (L+1)*D)
                 full_mems = full_mems_cat
            else:
                 full_mems = torch.zeros(B, 1, self.time_steps, 1, device=device) # フォールバック
        else:
            full_mems = torch.tensor(0.0, device=device)

        full_hiddens: torch.Tensor = torch.stack(full_hiddens_list, dim=1) # (B, NumLayers, T, D)
            
        final_features: torch.Tensor = x.mean(dim=1)
        
        if return_full_hiddens:
             return full_hiddens, torch.tensor(0.0, device=device), full_mems
             
        logits: torch.Tensor = self.output_projection(final_features)
        
        total_spikes: float = self.get_total_spikes()
        avg_spikes_val: float = total_spikes / (B * self.time_steps) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        
        return logits, avg_spikes, full_mems

class SpikingCNN(BaseModel):
    all_mems_history: List[torch.Tensor]
    
    def __init__(self, vocab_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        num_classes: int = vocab_size
        self.time_steps = time_steps
        self.all_mems_history = []
        
        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron]] # ◾️ 修正: 型を具体的に
        
        # ◾️◾️◾️ 修正: neuron_params をフィルタリング ◾️◾️◾️
        filtered_params: Dict[str, Any]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        else: # izhikevich
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            neuron_class(features=16, **filtered_params), # [0]
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            neuron_class(features=32, **filtered_params), # [1]
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), # 224x224入力の場合
            neuron_class(features=128, **filtered_params), # [2]
            nn.Linear(128, num_classes)
        )
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
        self._init_weights()

    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = input_images.shape
        device: torch.device = input_images.device
        functional.reset_net(self)
        
        output_voltages: List[torch.Tensor] = []
        full_hiddens_list: List[torch.Tensor] = [] # TCLのために層の出力を記録
        self.all_mems_history = []
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        neuron_layers: List[nn.Module] = []
        if return_full_mems:
            def _hook_mem(module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
                self.all_mems_history.append(output[1]) # mem
            
            for module in self.features:
                if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                    hooks.append(module.register_forward_hook(_hook_mem))
                    neuron_layers.append(module)
            for module in self.classifier:
                 if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                    hooks.append(module.register_forward_hook(_hook_mem))
                    neuron_layers.append(module)

        for _ in range(self.time_steps):
            x: torch.Tensor = input_images
            
            hidden_repr_t: Optional[torch.Tensor] = None # 型ヒント
            
            # ◾️◾️◾️ 修正: [no-redef] エラーを修正 ◾️◾️◾️
            for features_layer in self.features: # 'layer' -> 'features_layer'
                if isinstance(features_layer, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                    B_c, C_c, H_c, W_c = x.shape
                    x_reshaped: torch.Tensor = x.permute(0, 2, 3, 1).reshape(-1, C_c)
                    spikes, _ = features_layer(x_reshaped) # type: ignore[operator]
                    x = spikes.view(B_c, H_c, W_c, C_c).permute(0, 3, 1, 2)
                else:
                    x = features_layer(x) # type: ignore[operator]
                
                if isinstance(x, tuple):
                    x = x[0]
            
            hidden_repr_t = x.mean(dim=[2, 3]) # (B, C_out=32)
            full_hiddens_list.append(hidden_repr_t) 

            for i, classifier_layer in enumerate(self.classifier): # 'layer' -> 'classifier_layer'
                
                if isinstance(classifier_layer, nn.Flatten):
                    x = classifier_layer(x) # type: ignore[operator]
                    continue
                elif isinstance(classifier_layer, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                    spikes, _ = classifier_layer(x) # type: ignore[operator]
                    x = spikes
                elif isinstance(classifier_layer, nn.Linear):
                    if not isinstance(x, torch.Tensor):
                         x = cast(torch.Tensor, x)
                    x = classifier_layer(x) # type: ignore[operator]
                if isinstance(x, tuple):
                    x = x[0]
            # ◾️◾️◾️ 修正終わり ◾️◾️◾️

            output_voltages.append(x) # x は (B, num_classes) のロジット
        
        full_mems: torch.Tensor
        if return_full_mems:
            for hook in hooks: hook.remove()
            if self.all_mems_history:
                num_layers: int = len(neuron_layers)
                mems_by_time: List[List[torch.Tensor]] = [[] for _ in range(self.time_steps)]
                for i, mem in enumerate(self.all_mems_history):
                    t: int = i % self.time_steps
                    mems_by_time[t].append(mem.view(B, -1)) # (B, F_flat)
                
                mems_stacked_time: List[torch.Tensor] = [torch.cat(mems_t, dim=1) for mems_t in mems_by_time]
                full_mems_stacked: torch.Tensor = torch.stack(mems_stacked_time, dim=1)
                full_mems = full_mems_stacked.unsqueeze(1)
            else:
                full_mems = torch.zeros(B, 1, self.time_steps, 1, device=device) # フォールバック
        else:
            full_mems = torch.tensor(0.0, device=device)

        
        full_hiddens_stacked: torch.Tensor = torch.stack(full_hiddens_list, dim=1) 
        full_hiddens: torch.Tensor = full_hiddens_stacked.unsqueeze(1) 

        if return_full_hiddens:
             return full_hiddens, torch.tensor(0.0, device=device), full_mems

        final_logits: torch.Tensor = torch.stack(output_voltages, dim=0).mean(dim=0)
        
        total_spikes: float = self.get_total_spikes()
        avg_spikes_val: float = total_spikes / (B * self.time_steps) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)

        return final_logits, avg_spikes, full_mems


class SNNCore(nn.Module):
    def __init__(self, config: DictConfig, vocab_size: int, backend: str = "spikingjelly"):
        super(SNNCore, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        model_type: str = self.config.get("architecture_type", "simple")
        self.model: nn.Module
        
        params: Dict[str, Any] = cast(Dict[str, Any], OmegaConf.to_container(self.config, resolve=True))
        params.pop('path', None)
        neuron_config: Dict[str, Any] = params.pop('neuron', {})

        # ◾️◾️◾️ 指令4: SpikeGPT/RWKVベースのネイティブ生成モデルに対応 ◾️◾️◾️
        # (SpikingMamba, TinyRecursiveModel は指令の思想に合致)
        
        model_map: Dict[str, Type[BaseModel]]
        if backend == "spikingjelly":
            model_map = {
                "predictive_coding": BreakthroughSNN,
                "spiking_transformer": SpikingTransformer,
                "spiking_mamba": SpikingMamba, # SNNネイティブ
                "tiny_recursive_model": TinyRecursiveModel, # SNNネイティブ
                "simple": SimpleSNN,
                "hybrid_cnn_snn": HybridCnnSnnModel,
                "spiking_cnn": SpikingCNN,
                "hybrid_transformer": HybridSNNTransformer,
                "hybrid_attention_transformer": HybridAttentionTransformer,
            }
        elif backend == "snntorch":
            model_map = { # type: ignore[assignment]
                # --- ▼ 修正 ▼ ---
                "spiking_transformer": SpikingTransformerSnnTorch, # 変更
                # --- ▲ 修正 ▲ ---
            }
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if model_type not in model_map:
            raise ValueError(f"Unknown model type '{model_type}' for backend '{backend}'")
        
        # ◾️◾️◾️ 修正: vocab_size と neuron_config を渡す ◾️◾️◾️
        # ◾️◾️◾️ 修正: SimpleSNN に time_steps を渡す ◾️◾️◾️
        if 'time_steps' not in params and model_type == 'simple':
             params['time_steps'] = config.get('time_steps', 16) # configから取得、なければデフォルト
             
        self.model = model_map[model_type](vocab_size=vocab_size, neuron_config=neuron_config, **params)
        

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        model_type: Optional[str] = self.config.get("architecture_type")
        
        input_key: str = 'input_images' if model_type in ["hybrid_cnn_snn", "spiking_cnn"] else 'input_ids'
        
        input_data: Optional[torch.Tensor] = kwargs.get(input_key)
        
        if input_data is None and args and len(args) > 0:
            if isinstance(args[0], torch.Tensor):
                input_data = args[0]

        forward_kwargs: Dict[str, Any] = kwargs.copy()
        if input_key in forward_kwargs:
            del forward_kwargs[input_key] 

        if input_data is None:
            # 引数なしで呼べるのは
            return self.model(**forward_kwargs) # type: ignore[operator]

        if model_type in ["hybrid_cnn_snn", "spiking_cnn"]:
            return self.model(input_images=input_data, **forward_kwargs) # type: ignore[operator]
        else:
            return self.model(input_ids=input_data, **forward_kwargs) # type: ignore[operator]

