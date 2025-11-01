# ファイルパス: snn_research/core/__init__.py
# (修正)
# mypy [attr-defined] エラーを解消するため、
# snn_core.py よりも先に sntorch_models.py をインポートするよう
# 順序を変更。

from .base import BaseModel
# --- ▼ 修正: sntorch_models を先にインポート ▼ ---
from .sntorch_models import SpikingTransformerSnnTorch 
# --- ▲ 修正 ▲ ---
from .snn_core import SNNCore, BreakthroughSNN, SpikingTransformer, SimpleSNN
from .mamba_core import SpikingMamba
from .trm_core import TinyRecursiveModel
from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron
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
    "SpikingTransformerSnnTorch",
]
