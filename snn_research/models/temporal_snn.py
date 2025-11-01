# ファイルパス: snn_research/models/temporal_snn.py
# Title: 時系列データ特化 SNN モデル (RSNN実装)
# Description: Improvement-Plan.md に基づき、音声やセンサーデータなどの
#              時系列データ処理に特化したSNNモデルを実装します。
#              シンプルな再帰型SNN (RSNN) を AdaptiveLIFNeuron を使用して構築します。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Type, Optional
import math

# 既存のニューロンクラスをインポート
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from snn_research.core.base import BaseModel # BaseModelをインポートして継承

# spikingjellyのfunctionalをリセットに利用
from spikingjelly.activation_based import functional # type: ignore

class TemporalFeatureExtractor(BaseModel): # BaseModelを継承
    """
    時系列データ処理に特化したシンプルな再帰型SNN (RSNN) モデル。
    入力層 -> LIF -> 再帰層 -> LIF -> 出力層 の構造を持つ。
    """
    # 型ヒントを追加
    hidden_neuron: nn.Module
    output_neuron: Optional[nn.Module]

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, time_steps: int, neuron_config: Dict[str, Any], output_spikes: bool = True):
        """
        Args:
            input_dim (int): 入力特徴量の次元数。
            hidden_dim (int): 隠れ層（再帰層）のニューロン数。
            output_dim (int): 出力層の次元数。
            time_steps (int): 処理する時間ステップ数 (forward内で使用)。
            neuron_config (Dict[str, Any]): ニューロンの設定 (type, パラメータなど)。
            output_spikes (bool): 出力層もスパイクニューロンにするか。Falseの場合、Linear層のみ。
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_steps = time_steps # 内部処理ステップ数として保持
        self.output_spikes = output_spikes

        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)

        # 実際のニューロンクラスを選択
        neuron_class: Type[nn.Module]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        # 入力層 -> 隠れ層
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        # 隠れ層 -> 隠れ層 (再帰接続)
        self.recurrent = nn.Linear(hidden_dim, hidden_dim, bias=False) # 再帰接続はバイアスなしが一般的
        # 隠れ層ニューロン
        self.hidden_neuron = neuron_class(features=hidden_dim, **neuron_params)

        # 隠れ層 -> 出力層
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        # 出力層ニューロン (オプション)
        if self.output_spikes:
            self.output_neuron = neuron_class(features=output_dim, **neuron_params)
        else:
            self.output_neuron = None # 出力層がLinearのみの場合

        # 重みの初期化 (BaseModelから継承)
        self._init_weights()
        print(f"✅ TemporalFeatureExtractor (RSNN) initialized.")
        print(f"   - Input Dim: {input_dim}, Hidden Dim: {hidden_dim}, Output Dim: {output_dim}")
        print(f"   - Neuron Type: {neuron_type}, Output Spikes: {output_spikes}")

    def forward(self, input_sequence: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        時系列入力 (Batch, TimeSteps, InputDim) を処理します。
        BaseModelのインターフェースに合わせて (logits, avg_spikes, mem) を返します。

        Args:
            input_sequence (torch.Tensor): 入力時系列データ。
            return_spikes (bool): 平均スパイク数を計算するかどうか。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (最終的な出力 (ロジットまたはスパイクカウント), 平均スパイク数, 最終膜電位 (ダミー))
        """
        B, T_input, _ = input_sequence.shape
        device = input_sequence.device

        # ニューロンの状態をリセット
        functional.reset_net(self)

        # 隠れ状態の初期化
        # AdaptiveLIFNeuronなどは内部で状態を持つため、明示的なmem初期化は不要な場合がある
        # hidden_mem = torch.zeros(B, self.hidden_dim, device=device)
        hidden_spikes = torch.zeros(B, self.hidden_dim, device=device) # 前のステップのスパイクを保持
        # 出力状態の初期化
        # output_mem = torch.zeros(B, self.output_dim, device=device)

        output_history = [] # 全タイムステップの出力を記録

        # 時間ステップループ (入力シーケンス長 T_input でループ)
        for t in range(T_input):
            input_t = input_sequence[:, t, :] # (Batch, InputDim)

            # 隠れ層への入力 = 外部入力 + 前の時間ステップの隠れ層スパイクからの再帰入力
            recurrent_input = self.recurrent(hidden_spikes)
            hidden_input_current = self.input_to_hidden(input_t) + recurrent_input

            # 隠れ層ニューロンの更新
            # ニューロンは (spike, mem) を返す
            hidden_spikes, _ = self.hidden_neuron(hidden_input_current) # 新しいスパイクで上書き

            # 出力層への入力
            output_input_current = self.hidden_to_output(hidden_spikes)

            # 出力層の処理
            if self.output_neuron:
                output_t, _ = self.output_neuron(output_input_current)
            else:
                output_t = output_input_current # Linear層の出力がそのまま最終出力

            output_history.append(output_t)

        # 全タイムステップの出力をスタック (Batch, TimeSteps, OutputDim)
        final_output_sequence = torch.stack(output_history, dim=1)

        # SNN Coreのインターフェースに合わせる
        # ここでは、時間全体で平均化するか、最後のステップを使うか選択できる
        # 例: 最後のタイムステップの出力をロジットとする
        final_output = final_output_sequence[:, -1, :]

        # 平均スパイク数を計算 (BaseModelのメソッドを利用)
        avg_spikes_val = self.get_total_spikes() / (B * T_input) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=device)

        # 膜電位はここでは単純化して0を返す
        mem = torch.tensor(0.0, device=device)

        return final_output, avg_spikes, mem

    # reset メソッドは BaseModel から継承される reset_spike_stats を使用