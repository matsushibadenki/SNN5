# ファイルパス: snn_research/core/__init__.py
# (リファクタリング反映版)
# 修正: 削除対象の旧式モデル (v1, simple) への参照を削除

from .base import BaseModel, SNNLayerNorm

# --- ファクトリ ---
from .snn_core import SNNCore

# --- ニューロン ---
from .neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron,
    BistableIFNeuron
)

# --- SOTAアーキテクチャ ---
from .mamba_core import SpikingMamba
from .trm_core import TinyRecursiveModel
from .sntorch_models import SpikingTransformerSnnTorch 

# --- 分離されたローカルモデル ---
from .models.predictive_coding_model import BreakthroughSNN
# from .models.spiking_transformer_v1_model import SpikingTransformer_OldTextOnly # 削除
# from .models.simple_snn_model import SimpleSNN # 削除
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
    "BistableIFNeuron",
    
    # SOTAモデル
    "SpikingMamba",
    "TinyRecursiveModel",
    "SpikingTransformerSnnTorch",
    
    # ローカルモデル (SNNCoreが使用)
    "BreakthroughSNN",
    # "SpikingTransformer_OldTextOnly", # 削除
    # "SimpleSNN", # 削除
    "HybridCnnSnnModel",
    "SpikingCNN",
]