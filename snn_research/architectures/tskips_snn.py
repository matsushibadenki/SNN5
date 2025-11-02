# ファイルパス: snn_research/architectures/tskips_snn.py
# (新規作成)
#
# Title: TSkipsSNN (Temporal Skips SNN)
#
# Description:
# SNN5改善レポート (セクション6.2, 引用[115, 117]) に基づき、
# SHDデータセット等でSOTAを達成した「時間的遅延を持つスキップ接続 (TSkips)」
# の概念を実装するアーキテクチャ。
#
# この実装は、順方向 (FTSkips) および 逆方向 (BTSkips) の遅延接続を持つ
# 再帰的なSNNブロックをスタックするものです。
#
# mypy --strict 準拠。

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
        super().set_stateful(stateful)
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
        if self.forward_skip_buffer[0].shape[0] != B or self.forward_skip_buffer[0].device != x_t.device:
            self.forward_skip_buffer = [torch.zeros(B, self.features, device=x_t.device)] * (self.max_f_delay + 1)
        if self.backward_skip_buffer[0].shape[0] != B or self.backward_skip_buffer[0].device != x_t.device:
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
        current_input = self.fc1(x_t) + forward_skip_input + backward_skip_input
        
        spikes_t, _ = self.lif1(current_input) # (B, F)

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
    """
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        hidden_features: int,
        num_layers: int,
        time_steps: int,
        neuron_config: Dict[str, Any],
        forward_delays_per_layer: List[Optional[List[int]]],
        backward_delays_per_layer: List[Optional[List[int]]],
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[nn.Module]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            neuron_params = {k: v for k, v in neuron_params.items() if k in ['tau_mem', 'base_threshold']}
        else:
            neuron_class = IzhikevichNeuron
            neuron_params = {k: v for k, v in neuron_params.items() if k in ['a', 'b', 'c', 'd']}

        self.input_proj = nn.Linear(input_features, hidden_features)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TSkipsBlock(
                    features=hidden_features,
                    neuron_class=neuron_class,
                    neuron_params=neuron_params,
                    forward_delays=forward_delays_per_layer[i],
                    backward_delays=backward_delays_per_layer[i]
                )
            )
            
        self.output_proj = nn.Linear(hidden_features, num_classes)
        # 出力層はスパイクさせず、膜電位を平均化する
        self.output_lif = AdaptiveLIFNeuron(features=num_classes, **neuron_params)

        self._init_weights()
        logger.info(f"✅ TSkipsSNN initialized with {num_layers} layers.")

    def forward(
        self, 
        input_sequence: torch.Tensor, # (B, T, F_in)
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, T_seq, F_in = input_sequence.shape
        device: torch.device = input_sequence.device
        
        SJ_F.reset_net(self)
        
        # 各層の逆方向(B)スキップ接続の入力を保持するリスト
        # b_inputs[layer_idx][delay_idx]
        b_inputs: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        
        output_voltages: List[torch.Tensor] = []

        for t in range(T_seq):
            x_t: torch.Tensor = input_sequence[:, t, :] # (B, F_in)
            x_t = self.input_proj(x_t) # (B, F_hidden)
            
            # レイヤー間の信号伝播
            for i in range(self.num_layers):
                layer: TSkipsBlock = cast(TSkipsBlock, self.layers[i])
                
                # この層が受け取る逆方向スキップ入力を渡す
                current_b_inputs: List[torch.Tensor] = b_inputs[i]
                
                # 順伝播
                x_t, b_outputs = layer(x_t, current_b_inputs)
                
                # 次の層のための逆方向スキップ入力を準備
                if (i + 1) < self.num_layers:
                    b_inputs[i+1] = b_outputs
            
            # 最終層の出力
            final_output_current = self.output_proj(x_t)
            # 出力層のLIFで積分
            _, v_out_t = self.output_lif(final_output_current)
            output_voltages.append(v_out_t)

        # 時間全体で膜電位を平均化
        logits: torch.Tensor = torch.stack(output_voltages, dim=1).mean(dim=1) # (B, F_out)
        
        avg_spikes_val: float = self.get_total_spikes() / (B * T_seq) if return_spikes and T_seq > 0 else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        mem: torch.Tensor = torch.tensor(0.0, device=device)

        return logits, avg_spikes, mem
