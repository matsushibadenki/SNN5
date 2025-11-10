# ファイルパス: snn_research/io/__init__.py
# (更新)

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