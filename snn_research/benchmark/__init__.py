# ファイルパス: snn_research/benchmark/__init__.py
# Title: ベンチマークモジュール初期化
# Description: ベンチマーク関連のクラスや関数を公開し、タスクレジストリを定義する。
from typing import Dict, Type

from .ann_baseline import ANNBaselineModel
from .tasks import SST2Task, CIFAR10Task, BenchmarkTask, MRPCTask
from .metrics import calculate_accuracy

TASK_REGISTRY: Dict[str, Type[BenchmarkTask]] = {
    "sst2": SST2Task,
    "cifar10": CIFAR10Task,
    "mrpc": MRPCTask,
}

__all__ = ["ANNBaselineModel", "SST2Task", "CIFAR10Task", "MRPCTask", "calculate_accuracy", "TASK_REGISTRY", "BenchmarkTask"]