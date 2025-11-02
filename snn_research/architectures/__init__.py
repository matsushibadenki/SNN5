# ファイルパス: snn_research/architectures/__init__.py
# (更新)
# Title: SNN アーキテクチャ パッケージ
# Description:
# プロジェクトで使用される主要なSNNモデルアーキテクチャを定義する
# モジュール（v2 Transformer, Hybridなど）をインポートします。
# mypy --strict 準拠。
# 修正 (v2): SpikingDiffusionModel を __all__ に追加。
# 修正 (v3): SEWResNet を __all__ に追加。
#
# 追加 (v4): SNN5改善レポートに基づき、TSkipsSNN を追加。

from .spiking_transformer_v2 import SpikingTransformerV2, SDSAEncoderLayer
from .hybrid_attention_transformer import HybridAttentionTransformer, AdaptiveTransformerLayer
from .hybrid_neuron_network import HybridSpikingCNN
from .hybrid_transformer import HybridSNNTransformer, HybridTransformerLayer
from .spiking_diffusion_model import SpikingDiffusionModel
from .sew_resnet import SEWResNet, SEWResidualBlock
# --- ▼ 追加 ▼ ---
from .tskips_snn import TSkipsSNN, TSkipsBlock
# --- ▲ 追加 ▲ ---
from typing import List

__all__: List[str] = [
    "SpikingTransformerV2",
    "SDSAEncoderLayer",
    "HybridAttentionTransformer",
    "AdaptiveTransformerLayer",
    "HybridSpikingCNN",
    "HybridSNNTransformer",
    "HybridTransformerLayer",
    "SpikingDiffusionModel",
    "SEWResNet",
    "SEWResidualBlock",
    # --- ▼ 追加 ▼ ---
    "TSkipsSNN",
    "TSkipsBlock",
    # --- ▲ 追加 ▲ ---
]
