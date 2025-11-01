# ファイルパス: snn_research/architectures/__init__.py
# (新規作成)
# Title: SNN アーキテクチャ パッケージ
# Description:
# プロジェクトで使用される主要なSNNモデルアーキテクチャを定義する
# モジュール（v2 Transformer, Hybridなど）をインポートします。
# mypy --strict 準拠。

from .spiking_transformer_v2 import SpikingTransformerV2, SDSAEncoderLayer
from .hybrid_attention_transformer import HybridAttentionTransformer, AdaptiveTransformerLayer
from .hybrid_neuron_network import HybridSpikingCNN
from .hybrid_transformer import HybridSNNTransformer, HybridTransformerLayer
from typing import List

__all__: List[str] = [
    "SpikingTransformerV2",
    "SDSAEncoderLayer",
    "HybridAttentionTransformer",
    "AdaptiveTransformerLayer",
    "HybridSpikingCNN",
    "HybridSNNTransformer",
    "HybridTransformerLayer"
]