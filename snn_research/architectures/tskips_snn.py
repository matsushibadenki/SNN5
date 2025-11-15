# ファイルパス: snn_research/architectures/tskips_snn.py
# (HOPEアーキテクチャ対応)
#
# Title: TSkipsSNN (Temporal Skips SNN) - HOPE対応版
#
# Description:
# SNN5改善レポート (セクション6.2, 引用[115, 117]) に基づく TSkips の実装。
#
# --- 修正 (HOPE P1.3) ---
# doc/ROADMAP.md (Phase 4-1 / P1.3) に基づき、
# HOPE (Hierarchical Temporal Architecture) の概念を導入。
# TSkipsBlock のスタックを「モチーフ層」と「構文層」に分離し、
# 階層的な時系列処理を実現する。
#
# mypy --strict 準拠。
#
# 修正 (v_hpo_fix_attribute_error):
# - AttributeError: 'super' object has no attribute 'set_stateful' を修正。
# - super().set_stateful(stateful) を self.stateful = stateful に変更。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Type, Optional, cast

from snn_research.core.base import BaseModel
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from spikingjelly.activation_based import functional as SJ_F # type: ignore
from spikingjelly.activation_based import base as sj_base # type: ignore

import logging
logger = logging.getLogger(__name__)

class TSkipsBlock(sj_base.MemoryModule):
    """
    時間的遅延接続 (TSkips) を持つSNNブロック。
    引用[115]に基づく。
    (HOPE P1.3: このブロックは変更なしで流用)
    """
    lif1: nn.Module
    fc1: nn.Linear
    
    # 時間的遅延接続用のバッファ
    forward_skip_buffer: List[torch.Tensor]
    backward_skip_buffer: List[torch.Tensor]

    def __init__(
        self,
        features: int,
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any],
        forward_delays: Optional[List[int]] = None, # 例: [1, 3] (1ステップ先、3ステップ先へ)
        backward_delays: Optional[List[int]] = None # 例: [1] (1ステップ前の層から)
    ):
        super().__init__()
        self.features = features
        self.fc1 = nn.Linear(features, features)
        self.lif1 = neuron_class(features=features, **neuron_params)
        
        self.forward_delays = sorted(forward_delays) if forward_delays else []
        self.backward_delays = sorted(backward_delays) if backward_delays else []
        
        self.max_f_delay = max(self.forward_delays) if self.forward_delays else 0
        self.max_b_delay = max(self.backward_delays) if self.backward_delays else 0
        
        # 遅延接続用の重み (学習可能)
        if self.forward_delays:
            self.f_skip_weights = nn.Parameter(torch.randn(len(self.forward_delays), features) * 0.1)
        if self.backward_delays:
            self.b_skip_weights = nn.Parameter(torch.randn(len(self.backward_delays), features) * 0.1)

        self.reset_buffers()
        
    def set_stateful(self, stateful: bool) -> None:
        # --- ▼ 修正 (v_hpo_fix_attribute_error) ▼ ---
        # super().set_stateful(stateful) # 誤り
        self.stateful = stateful # 正しい
        # --- ▲ 修正 (v_hpo_fix_attribute_error) ▲ ---
        if hasattr(self.lif1, 'set_stateful'):
            cast(Any, self.lif1).set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        if hasattr(self.lif1, 'reset'):
            cast(Any, self.lif1).reset()
        self.reset_buffers()
        
    def reset_buffers(self) -> None:
        """遅延接続バッファをリセットする"""
        # (B, F) の形状を想定
        dummy_tensor = torch.zeros(1, self.features) 
        self.forward_skip_buffer = [dummy_tensor] * (self.max_f_delay + 1)
        self.backward_skip_buffer = [dummy_tensor] * (self.max_b_delay + 1)

    def forward(
        self, 
        x_t: torch.Tensor, 
        backward_inputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        1タイムステップ分の処理。
        
        Args:
            x_t (torch.Tensor): 現在のタイムステップの入力 (B, F)
            backward_inputs (List[torch.Tensor]): 
                過去の層からの逆方向遅延入力 (遅延ごとにリスト化)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                (現在の出力スパイク (B, F), 
                 未来の層への順方向遅延入力 (遅延ごとにリスト化))
        """
        B = x_t.shape[0]
        # (mypy互換性) バッファがNoneでないか、または形状/デバイスが異なるかチェック
        if not self.forward_skip_buffer or self.forward_skip_buffer[0].shape[0] != B or self.forward_skip_buffer[0].device != x_t.device:
            self.forward_skip_buffer = [torch.zeros(B, self.features, device=x_t.device)] * (self.max_f_delay + 1)
        if not self.backward_skip_buffer or self.backward_skip_buffer[0].shape[0] != B or self.backward_skip_buffer[0].device != x_t.device:
            self.backward_skip_buffer = [torch.zeros(B, self.features, device=x_t.device)] * (self.max_b_delay + 1)


        # --- 1. 順方向 (Forward) スキップ接続の入力を計算 ---
        # バッファから指定された遅延のスパイクを取り出す
        forward_skip_input = torch.zeros_like(x_t)
        if self.forward_delays:
            for i, delay in enumerate(self.forward_delays):
                forward_skip_input += self.forward_skip_buffer[delay] * self.f_skip_weights[i]

        # --- 2. 逆方向 (Backward) スキップ接続の入力を計算 ---
        backward_skip_input = torch.zeros_like(x_t)
        if self.backward_delays and backward_inputs:
            for i, delay in enumerate(self.backward_delays):
                if i < len(backward_inputs):
                    # backward_inputs[i] は、i番目の遅延に対応する入力
                    backward_skip_input += backward_inputs[i] * self.b_skip_weights[i]

        # --- 3. メインパスの計算 ---
        # 入力 = 現在の入力 + 順方向スキップ + 逆方向スキップ
        # (mypy互換性) self.lif1 は nn.Module であり callable
        lif_callable = cast(nn.Module, self.lif1)
        current_input = self.fc1(x_t) + forward_skip_input + backward_skip_input
        
        spikes_t_tuple = lif_callable(current_input) # (B, F)
        
        # (mypy互換性) ニューロンがタプル(spikes, mem)を返すことを想定
        if isinstance(spikes_t_tuple, tuple):
            spikes_t = spikes_t_tuple[0]
        else:
            spikes_t = spikes_t_tuple

        # --- 4. バッファの更新 ---
        # 順方向バッファを更新 (未来の自分自身/層のため)
        self.forward_skip_buffer.insert(0, spikes_t)
        self.forward_skip_buffer.pop()
        
        # 逆方向バッファを更新 (未来の次の層のため)
        self.backward_skip_buffer.insert(0, spikes_t)
        self.backward_skip_buffer.pop()

        # --- 5. 出力の準備 ---
        # 次の層に渡すための、この層からの逆方向遅延出力
        backward_outputs = [self.backward_skip_buffer[delay] for delay in self.backward_delays]
        
        return spikes_t, backward_outputs

class TSkipsSNN(BaseModel):
    """
    TSkipsBlockを複数層スタックしたSNNモデル。
    SHD (Spiking Heidelberg Digits) などの時系列データ処理を想定。
    
    --- 修正 (HOPE P1.3) ---
    モチーフ層と構文層の階層分離を導入。
    """
    
    # --- ▼ 修正 (HOPE P1.3) ▼ ---
    motif_layers: nn.ModuleList
    syntax_layers: nn.ModuleList
    motif_to_syntax_proj: nn.Linear
    # --- ▲ 修正 (HOPE P1.3) ▲ ---

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        hidden_features: int,
        # num_layers: int, # (HOPE P1.3: 分離する)
        time_steps: int,
        neuron_config: Dict[str, Any],
        # (HOPE P1.3: 層定義を分離)
        num_motif_layers: int,
        num_syntax_layers: int,
        motif_forward_delays: List[Optional[List[int]]],
        motif_backward_delays: List[Optional[List[int]]],
        syntax_forward_delays: List[Optional[List[int]]],
        syntax_backward_delays: List[Optional[List[int]]],
        # (HOPE P1.3: TSkipsSNN.yaml から古い引数を削除したため、
        #  forward_delays_per_layer と backward_delays_per_layer の
        #  引数を削除)
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs) # (mypy互換性) kwargs を BaseModel に渡す
        self.time_steps = time_steps
        
        # --- ▼ 修正 (HOPE P1.3): 層数の検証 ▼ ---
        self.num_motif_layers = num_motif_layers
        self.num_syntax_layers = num_syntax_layers
        self.num_layers = num_motif_layers + num_syntax_layers # 合計層数
        
        if len(motif_forward_delays) != num_motif_layers or len(motif_backward_delays) != num_motif_layers:
            raise ValueError(f"Motif delay definitions ({len(motif_forward_delays)}) must match num_motif_layers ({num_motif_layers})")
        if len(syntax_forward_delays) != num_syntax_layers or len(syntax_backward_delays) != num_syntax_layers:
            raise ValueError(f"Syntax delay definitions ({len(syntax_forward_delays)}) must match num_syntax_layers ({num_syntax_layers})")
        # --- ▲ 修正 (HOPE P1.3) ▲ ---


        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[nn.Module]
        
        # (mypy互換性) AdaptiveLIFNeuron が期待するパラメータのみを渡す
        lif_params = {k: v for k, v in neuron_params.items() if k in ['threshold', 'decay', 'bias_init', 'v_init', 'time_steps']}
        
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            # (mypy互換性) AdaptiveLIFNeuron が期待するパラメータのみを渡す
            neuron_params_cleaned = lif_params
        else:
            neuron_class = IzhikevichNeuron
            neuron_params_cleaned = {k: v for k, v in neuron_params.items() if k in ['a', 'b', 'c', 'd', 'v_init']}

        self.input_proj = nn.Linear(input_features, hidden_features)
        
        # --- ▼ 修正 (HOPE P1.3): モチーフ層と構文層の分離 ▼ ---
        self.motif_layers = nn.ModuleList()
        for i in range(self.num_motif_layers):
            self.motif_layers.append(
                TSkipsBlock(
                    features=hidden_features,
                    neuron_class=neuron_class,
                    neuron_params=neuron_params_cleaned,
                    forward_delays=motif_forward_delays[i],
                    backward_delays=motif_backward_delays[i]
                )
            )
            
        # モチーフ層から構文層への接続 (次元は同じ)
        self.motif_to_syntax_proj = nn.Linear(hidden_features, hidden_features)
        
        self.syntax_layers = nn.ModuleList()
        for i in range(self.num_syntax_layers):
            self.syntax_layers.append(
                TSkipsBlock(
                    features=hidden_features,
                    neuron_class=neuron_class,
                    neuron_params=neuron_params_cleaned,
                    forward_delays=syntax_forward_delays[i],
                    backward_delays=syntax_backward_delays[i]
                )
            )
        # --- ▲ 修正 (HOPE P1.3) ▲ ---
            
        self.output_proj = nn.Linear(hidden_features, num_classes)
        # 出力層はスパイクさせず、膜電位を平均化する
        self.output_lif = AdaptiveLIFNeuron(features=num_classes, **lif_params)

        self._init_weights()
        logger.info(f"✅ HOPE-TSkipsSNN initialized ({self.num_motif_layers} Motif + {self.num_syntax_layers} Syntax layers).")

    def forward(
        self, 
        input_sequence: torch.Tensor, # (B, T, F_in)
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, T_seq, F_in = input_sequence.shape
        if T_seq == 0:
             logger.warning("TSkipsSNN received empty sequence (T=0). Returning zeros.")
             logits = torch.zeros(B, cast(int, self.output_proj.out_features), device=input_sequence.device)
             avg_spikes = torch.tensor(0.0, device=input_sequence.device)
             mem = torch.tensor(0.0, device=input_sequence.device)
             return logits, avg_spikes, mem
             
        device: torch.device = input_sequence.device
        
        SJ_F.reset_net(self)
        
        # --- ▼ 修正 (HOPE P1.3): 階層分離 ▼ ---
        
        # 各層の逆方向(B)スキップ接続の入力を保持するリスト
        # b_inputs[layer_idx][delay_idx]
        motif_b_inputs: List[List[torch.Tensor]] = [[] for _ in range(self.num_motif_layers)]
        syntax_b_inputs: List[List[torch.Tensor]] = [[] for _ in range(self.num_syntax_layers)]
        
        output_voltages: List[torch.Tensor] = []
        
        # (HOPE P1.3) 構文層は低頻度で実行する (例: 4ステップごと)
        # T_syntax_step = 4 
        # (簡易実装: まずは全ステップで実行し、接続のみ分離する)

        for t in range(T_seq):
            x_t: torch.Tensor = input_sequence[:, t, :] # (B, F_in)
            x_t = self.input_proj(x_t) # (B, F_hidden)
            
            # --- 1. モチーフ層 (短時間処理) ---
            x_motif = x_t
            for i in range(self.num_motif_layers):
                layer: TSkipsBlock = cast(TSkipsBlock, self.motif_layers[i])
                current_b_inputs: List[torch.Tensor] = motif_b_inputs[i]
                
                x_motif, b_outputs = layer(x_motif, current_b_inputs)
                
                if (i + 1) < self.num_motif_layers:
                    motif_b_inputs[i+1] = b_outputs
            
            # --- 2. 構文層への入力 ---
            x_syntax = self.motif_to_syntax_proj(x_motif)
            
            # (HOPE P1.3) ここで構文層の低頻度実行
            # if t % T_syntax_step == 0:
            
            # --- 3. 構文層 (長時間処理) ---
            for i in range(self.num_syntax_layers):
                layer: TSkipsBlock = cast(TSkipsBlock, self.syntax_layers[i])
                
                # 構文層の最初の層は、モチーフ層の最後の層からの
                # 逆方向(B)接続を受け取る (もしあれば)
                if i == 0:
                    current_b_inputs = b_outputs # モチーフ層の最後の b_outputs
                else:
                    current_b_inputs = syntax_b_inputs[i]
                
                x_syntax, b_outputs = layer(x_syntax, current_b_inputs)
                
                if (i + 1) < self.num_syntax_layers:
                    syntax_b_inputs[i+1] = b_outputs
            
            # --- 4. 最終層の出力 ---
            final_output_current = self.output_proj(x_syntax)
            
            # (mypy互換性) self.output_lif は nn.Module であり callable
            output_lif_callable = cast(nn.Module, self.output_lif)
            output_tuple = output_lif_callable(final_output_current)
            
            if isinstance(output_tuple, tuple):
                _, v_out_t = output_tuple
            else:
                # (もしv_memを返さないニューロンだった場合 - ここでは起こらないはず)
                v_out_t = output_tuple 
                
            output_voltages.append(v_out_t)
        # --- ▲ 修正 (HOPE P1.3) ▲ ---

        # 時間全体で膜電位を平均化
        logits: torch.Tensor = torch.stack(output_voltages, dim=1).mean(dim=1) # (B, F_out)
        
        # (mypy互換性) T_seq が 0 でないことは上でチェック済み
        avg_spikes_val: float = self.get_total_spikes() / (B * T_seq) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        mem: torch.Tensor = torch.tensor(0.0, device=device)

        return logits, avg_spikes, mem
