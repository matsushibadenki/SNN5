# ファイルパス: snn_research/core/__init__.py
# (リファクタリング反映版)

from .base import BaseModel, SNNLayerNorm

# --- ファクトリ ---
from .snn_core import SNNCore

# --- ニューロン ---
from .neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron,
    BistableIFNeuron # <-- 修正: BistableIFNeuron を追加
)

# --- SOTAアーキテクチャ ---
from .mamba_core import SpikingMamba
from .trm_core import TinyRecursiveModel
from .sntorch_models import SpikingTransformerSnnTorch 

# --- 分離されたローカルモデル ---
from .models.predictive_coding_model import BreakthroughSNN
from .models.spiking_transformer_v1_model import SpikingTransformer_OldTextOnly
from .models.simple_snn_model import SimpleSNN
from .models.hybrid_cnn_snn_model import HybridCnnSnnModel
from .models.spiking_cnn_model import SpikingCNN

__all__ = [
    "BaseModel",
    "SNNLayerNorm",
    "SNNCore",
    
    # ニューロン
    "AdaptiveLIFNeuron",
    "IzhikevichNeuron",
    "GLIFNeuron",
    "TC_LIF", 
    "DualThresholdNeuron", 
    "ScaleAndFireNeuron",
    "BistableIFNeuron", # <-- 修正: __all__ に追加
    
    # SOTAモデル
    "SpikingMamba",
    "TinyRecursiveModel",
    "SpikingTransformerSnnTorch",
    
    # ローカルモデル (SNNCoreが使用)
    "BreakthroughSNN",
    "SpikingTransformer_OldTextOnly",
    "SimpleSNN",
    "HybridCnnSnnModel",
    "SpikingCNN",
]
