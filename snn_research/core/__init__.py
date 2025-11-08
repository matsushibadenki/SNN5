# ファイルパス: snn_research/core/__init__.py
# (更新)
# mypy [attr-defined] エラーを解消するため、
# snn_core.py よりも先に sntorch_models.py をインポートするよう
# 順序を変更。
#
# 修正 (v2): GLIFNeuron を __all__ に追加。
#
# 追加 (v3): SNN5改善レポートに基づき、TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron を追加。
#
# 修正 (v_hpo_fix_importerror):
# - snn_core.py 内で SpikingTransformer が SpikingTransformer_OldTextOnly に
#   リネームされたため、インポート時にエイリアス(as)を使用して互換性を維持する。

from .base import BaseModel
from .sntorch_models import SpikingTransformerSnnTorch 
# --- ▼ 修正 (v_hpo_fix_importerror) ▼ ---
from .snn_core import SNNCore, BreakthroughSNN, SpikingTransformer_OldTextOnly as SpikingTransformer, SimpleSNN
# --- ▲ 修正 (v_hpo_fix_importerror) ▲ ---
from .mamba_core import SpikingMamba
from .trm_core import TinyRecursiveModel
# --- ▼ 修正 ▼ ---
from .neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron # SNN5改善レポートによる追加
)
# --- ▲ 修正 ▲ ---
# from .sntorch_models import SpikingTransformerSnnTorch # <-- 元の位置

__all__ = [
    "BaseModel",
    "SNNCore",
    "BreakthroughSNN",
    "SpikingTransformer", # エイリアスにより、この名前でのエクスポートが引き続き機能する
    "SpikingMamba",
    "TinyRecursiveModel",
    "SimpleSNN",
    "AdaptiveLIFNeuron",
    "IzhikevichNeuron",
    # --- ▼ 修正 ▼ ---
    "GLIFNeuron",
    "TC_LIF", 
    "DualThresholdNeuron", 
    "ScaleAndFireNeuron",
    # --- ▲ 修正 ▲ ---
    "SpikingTransformerSnnTorch",
]
