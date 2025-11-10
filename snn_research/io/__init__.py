# ファイルパス: snn_research/io/__init__.py
# (更新)
#
# Title: IO（入出力）パッケージ
#
# Description:
# - 人工脳アーキテクチャの入出力（感覚受容、符号化、復号化、実行）を担うモジュール。
#
# 改善 (v4):
# - doc/ROADMAP.md (P2.2) に基づき、FrequencyEncoder を __all__ に追加。

from .sensory_receptor import SensoryReceptor
# --- ▼ 修正 ▼ ---
from .spike_encoder import SpikeEncoder, DifferentiableTTFSEncoder, FrequencyEncoder
from .spike_decoder import SpikeDecoder
from .actuator import Actuator
# --- ▲ 修正 ▲ ---

__all__ = [
    "SensoryReceptor",
    "SpikeEncoder",
    # --- ▼ 修正 ▼ ---
    "DifferentiableTTFSEncoder",
    "FrequencyEncoder", # P2.2 (FE) を公開
    "SpikeDecoder",
    "Actuator"
    # --- ▲ 修正 ▲ ---
]