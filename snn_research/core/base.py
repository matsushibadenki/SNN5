# ファイルパス: snn_research/core/base.py
# (新規作成)
# Title: SNNモデル 基底クラス
# Description:
# - 循環インポートを解消するため、複数のモデルアーキテクチャで共有される
#   基底クラス(BaseModel)と共通レイヤー(SNNLayerNorm)をこのファイルに分離する。
#
# --- 修正 (mypy) ---
# 1. __init__ が **kwargs を受け取るように修正 (call-arg エラー解消)
# 2. time_steps, total_spikes, _total_neurons を初期化 (has-type エラー解消)
# 3. .neurons からの循環インポートを削除
# 4. get_total_spikes/reset_spike_stats を spikingjelly の base.MemoryModule を
#    使うように汎用化 (循環インポート解消)
#
# --- 修正 (mypy v2) ---
# 1. [syntax] Unmatched '}' エラーを修正。
#    クラス定義の末尾に不要な '}' があったため削除。

import torch
import torch.nn as nn
from typing import Dict, Any

# --- ▼ 修正 (mypy) ▼ ---
# from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron # 循環インポートのため削除
from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]
# --- ▲ 修正 (mypy) ▲ ---


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
    # --- ▼ 修正 (mypy): __init__ 修正 ▼ ---
    def __init__(self, **kwargs: Any):
        super().__init__()
        # adaptive_lif_neuron.py や snn_core.py との互換性のために time_steps を受け取る
        self.time_steps: int = int(kwargs.get('time_steps', 0)) 
        # get_total_spikes / adaptive_lif_neuron.py の [has-type] エラー解消
        self.total_spikes: torch.Tensor = torch.tensor(0.0) 
        self._total_neurons: int = 0
    # --- ▲ 修正 (mypy) ▲ ---

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
        # --- ▼ 修正 (mypy): 汎用化 ▼ ---
        for module in self.modules():
            # spikingjelly の MemoryModule を継承し、
            # "total_spikes" 属性を持つモジュールをチェック
            if isinstance(module, sj_base.MemoryModule) and hasattr(module, 'total_spikes'):
                spikes = getattr(module, 'total_spikes')
                if isinstance(spikes, torch.Tensor):
                    total += spikes.item()
        # --- ▲ 修正 (mypy) ▲ ---
        return total
    
    def reset_spike_stats(self):
        """スパイク関連の統計情報をリセットする。"""
        # --- ▼ 修正 (mypy): 汎用化 ▼ ---
        for module in self.modules():
            # spikingjelly の MemoryModule の reset を呼ぶ
            if isinstance(module, sj_base.MemoryModule):
                module.reset()
        # --- ▲ 修正 (mypy) ▲ ---
