# snn_research/training/__init__.py

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
from .pruning import apply_magnitude_pruning

__all__ = [
    "BreakthroughTrainer", "DistillationTrainer", "SelfSupervisedTrainer",
    "PhysicsInformedTrainer", "ProbabilisticEnsembleTrainer", "ParticleFilterTrainer",
    "PlannerTrainer", "BPTTTrainer",
    "CombinedLoss", "DistillationLoss", "SelfSupervisedLoss", "PhysicsInformedLoss",
    "PlannerLoss", "ProbabilisticEnsembleLoss",
    "apply_qat", "convert_to_quantized_model",
    "apply_magnitude_pruning"
]