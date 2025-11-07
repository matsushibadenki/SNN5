# ファイルパス: snn_research/training/__init__.py
# (更新)

from .trainers import (
    BreakthroughTrainer, DistillationTrainer, SelfSupervisedTrainer,
    PhysicsInformedTrainer, ProbabilisticEnsembleTrainer, ParticleFilterTrainer,
    PlannerTrainer, BPTTTrainer
)
from .losses import (
    CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss,
    PlannerLoss, ProbabilisticEnsembleLoss
)
from .quantization import apply_qat, convert_to_quantized_model
# --- ▼ 修正 ▼ ---
from .pruning import apply_sbc_pruning, apply_spatio_temporal_pruning # apply_spatio_temporal_pruning を追加
# --- ▲ 修正 ▲ ---

__all__ = [
    "BreakthroughTrainer", "DistillationTrainer", "SelfSupervisedTrainer",
    "PhysicsInformedTrainer", "ProbabilisticEnsembleTrainer", "ParticleFilterTrainer",
    "PlannerTrainer", "BPTTTrainer",
    "CombinedLoss", "DistillationLoss", "SelfSupervisedLoss", "PhysicsInformedLoss",
    "PlannerLoss", "ProbabilisticEnsembleLoss",
    "apply_qat", "convert_to_quantized_model",
    # --- ▼ 修正 ▼ ---
    "apply_sbc_pruning",
    "apply_spatio_temporal_pruning" # 公開リストに追加
    # --- ▲ 修正 ▲ ---
]