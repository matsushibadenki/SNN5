# ファイルパス: snn_research/hybrid/adapter.py
# (新規作成)
# Title: ANN-SNN アダプタ層
# Description:
# doc/SNN開発：Gemma3:GPT-4のSNN変換に関する技術的考察と戦略.md (戦略B, 5.2) に基づき、
# ANN（アナログ）ドメインとSNN（スパイク）ドメイン間の情報変換を担う
# アダプタ層（AnalogToSpikes, SpikesToAnalog）を実装します。
# mypy --strict 準拠。

import torch
import torch.nn as nn
from typing import Type, Dict, Any, cast, List, Tuple

from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from spikingjelly.activation_based import functional as F # type: ignore
from snn_research.core.base import BaseModel # BaseModelをインポート (set_statefulのため)

class AnalogToSpikes(BaseModel): # BaseModelを継承
    """
    ANNのアナログ出力をSNNのスパイク入力に変換するアダプタ。
    単純なレートコーディング（アナログ値を電流としてLIFに入力）を実装。
    
    doc/SNN開発：Gemma3:GPT-4のSNN変換に関する技術的考察と戦略.md (5.2) の
    AnalogToSpikes (レートコーディング) に対応。
    """
    neuron: nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        time_steps: int,
        neuron_config: Dict[str, Any]
    ) -> None:
        """
        Args:
            in_features (int): 入力アナログ特徴量の次元数。
            out_features (int): 出力スパイク特徴量の次元数。
            time_steps (int): SNNの処理時間ステップ数。
            neuron_config (Dict[str, Any]): スパイク生成に使用するニューロンの設定。
        """
        super().__init__() # BaseModelの__init__を呼ぶ
        self.in_features = in_features
        self.out_features = out_features
        self.time_steps = time_steps
        
        # アナログ特徴量をSNNの入力次元に射影する線形層
        self.projection = nn.Linear(in_features, out_features)
        
        # スパイク生成ニューロン
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[nn.Module]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            # AdaptiveLIFNeuronが受け取るパラメータのみフィルタリング
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        else:
            raise ValueError(f"Unknown neuron type for AnalogToSpikes: {neuron_type_str}")
        
        self.neuron = neuron_class(features=out_features, **neuron_params)

    def forward(self, x_analog: torch.Tensor) -> torch.Tensor:
        """
        アナログテンソル (B, L, D_in) または (B, D_in) を
        スパイクテンソル (B, L, T, D_out) または (B, T, D_out) に変換する。

        Args:
            x_analog (torch.Tensor): ANNからのアナログ出力。

        Returns:
            torch.Tensor: SNNに入力するためのスパイク列。
        """
        # 1. 次元射影
        x_projected: torch.Tensor = self.projection(x_analog)
        
        # 2. 時間軸の導入
        # (B, L, D_out) -> (B, L, T, D_out)
        # (B, D_out) -> (B, T, D_out)
        # unsqueeze(-2) で時間次元を追加し、time_steps回リピート
        x_repeated: torch.Tensor = x_projected.unsqueeze(-2).repeat(1, *([1] * (x_analog.dim() - 1)), self.time_steps, 1)
        
        original_shape: Tuple[int, ...] = x_repeated.shape
        # (B * L * T, D_out) または (B * T, D_out) にフラット化
        x_flat: torch.Tensor = x_repeated.reshape(-1, self.out_features)

        # 3. スパイク生成
        F.reset_net(self) # 状態をリセット
        
        # --- 正確な時間ループによる実装 ---
        spikes_history: List[torch.Tensor] = []
        
        # (B * L, T, D_out) または (B, T, D_out) にリシェイプ
        x_time_batched: torch.Tensor = x_repeated.reshape(-1, self.time_steps, self.out_features)
        
        # ニューロンをStatefulに設定
        neuron_module: nn.Module = cast(nn.Module, self.neuron)
        if hasattr(neuron_module, 'set_stateful'):
            getattr(neuron_module, 'set_stateful')(True)

        for t in range(self.time_steps):
            # (B*L, D_out) または (B, D_out) の電流を入力
            current_input: torch.Tensor = x_time_batched[:, t, :]
            
            # ニューロンが (spike, mem) を返すと仮定
            spike_t, _ = self.neuron(current_input) 
            spikes_history.append(spike_t)
            
        # Statefulを解除
        if hasattr(neuron_module, 'set_stateful'):
            getattr(neuron_module, 'set_stateful')(False)

        # スパイクを (B*L, T, D_out) または (B, T, D_out) にスタック
        spikes_stacked: torch.Tensor = torch.stack(spikes_history, dim=1)
        
        # 元の形状 (B, L, T, D_out) または (B, T, D_out) に戻す
        output_shape: Tuple[int, ...]
        if x_analog.dim() == 3: # (B, L, D_in)
            output_shape = (original_shape[0], original_shape[1], self.time_steps, self.out_features)
        else: # (B, D_in)
            output_shape = (original_shape[0], self.time_steps, self.out_features)
            
        # (B, L, T, D_out) または (B, T, D_out) の形状で返す
        return spikes_stacked.reshape(output_shape)


class SpikesToAnalog(nn.Module):
    """
    SNNのスパイク出力をANNのアナログ入力に変換するアダプタ。
    時間平均（レートの計算）または最終膜電位の読み出しを行う。
    
    doc/SNN開発：Gemma3:GPT-4のSNN変換に関する技術的考察と戦略.md (5.2) の
    SpikesToAnalog (アグリゲータ) に対応。
    """
    def __init__(self, in_features: int, out_features: int, method: str = "rate") -> None:
        """
        Args:
            in_features (int): 入力スパイク特徴量の次元数。
            out_features (int): 出力アナログ特徴量の次元数。
            method (str): 集約方法。"rate" (時間平均) または "mem" (膜電位: 未実装)。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        
        # スパイク集約結果をアナログ特徴量に射影する線形層
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x_spikes: torch.Tensor) -> torch.Tensor:
        """
        スパイクテンソル (B, L, T, D_in) または (B, T, D_in) を
        アナログテンソル (B, L, D_out) または (B, D_out) に変換する。

        Args:
            x_spikes (torch.Tensor): SNNからのスパイク出力。

        Returns:
            torch.Tensor: ANNに入力するためのアナログ特徴量。
        """
        
        if self.method == "rate":
            # 時間次元 (-2) で平均を取り、発火率を計算
            x_aggregated: torch.Tensor = x_spikes.mean(dim=-2)
        else:
            raise NotImplementedError(f"Aggregation method '{self.method}' is not implemented.")
            
        # 射影してアナログ特徴量に変換
        x_analog: torch.Tensor = self.projection(x_aggregated)
        
        return x_analog