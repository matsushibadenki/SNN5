# ファイルパス: snn_research/core/__init__.py
# (更新)
# mypy [attr-defined] エラーを解消するため、
# snn_core.py よりも先に sntorch_models.py をインポートするよう
# 順序を変更。
#
# 修正 (v2): GLIFNeuron を __all__ に追加。

from .base import BaseModel
from .sntorch_models import SpikingTransformerSnnTorch 
from .snn_core import SNNCore, BreakthroughSNN, SpikingTransformer, SimpleSNN
from .mamba_core import SpikingMamba
from .trm_core import TinyRecursiveModel
# --- ▼ 修正 ▼ ---
from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron
# --- ▲ 修正 ▲ ---
# from .sntorch_models import SpikingTransformerSnnTorch # <-- 元の位置

__all__ = [
    "BaseModel",
    "SNNCore",
    "BreakthroughSNN",
    "SpikingTransformer",
    "SpikingMamba",
    "TinyRecursiveModel",
    "SimpleSNN",
    "AdaptiveLIFNeuron",
    "IzhikevichNeuron",
    # --- ▼ 修正 ▼ ---
    "GLIFNeuron",
    # --- ▲ 修正 ▲ ---
    "SpikingTransformerSnnTorch",
]
