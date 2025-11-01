# ファイルパス: snn_research/core/base.py
# (新規作成)
# Title: SNNモデル 基底クラス
# Description:
# - 循環インポートを解消するため、複数のモデルアーキテクチャで共有される
#   基底クラス(BaseModel)と共通レイヤー(SNNLayerNorm)をこのファイルに分離する。

import torch
import torch.nn as nn
from typing import Dict, Any

from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron

class SNNLayerNorm(nn.Module):
    """
    SNN用のLayerNorm。snn_core.pyから移動。
    """
    def __init__(self, normalized_shape: Any):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class BaseModel(nn.Module):
    """
    すべてのSNNモデルが継承する基底クラス。snn_core.pyから移動。
    重みの初期化やスパイク統計の共通メソッドを提供する。
    """
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def get_total_spikes(self) -> float:
        """モデル全体の総スパイク数を計算する。"""
        total = 0.0
        for module in self.modules():
            if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                total += module.total_spikes.item()
        return total
    
    def reset_spike_stats(self):
        """スパイク関連の統計情報をリセットする。"""
        for module in self.modules():
            if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                module.reset()

