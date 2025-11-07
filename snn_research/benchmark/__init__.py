# ファイルパス: snn_research/benchmark/__init__.py
# Title: ベンチマークモジュール初期化
# Description: ベンチマーク関連のクラスや関数を公開し、タスクレジストリを定義する。
#
# 修正 (v2): CIFAR10DVSTask をレジストリと __all__ に追加。
#
# 改善 (v3): SHDTask をレジストリと __all__ に追加。

from typing import Dict, Type

from .ann_baseline import ANNBaselineModel
# --- ▼ 修正 ▼ ---
from .tasks import SST2Task, CIFAR10Task, BenchmarkTask, MRPCTask, CIFAR10DVSTask, SHDTask
# --- ▲ 修正 ▲ ---
from .metrics import calculate_accuracy

TASK_REGISTRY: Dict[str, Type[BenchmarkTask]] = {
    "sst2": SST2Task,
    "cifar10": CIFAR10Task,
    "mrpc": MRPCTask,
    # --- ▼ 修正 ▼ ---
    "cifar10_dvs": CIFAR10DVSTask,
    "shd": SHDTask, # SHDタスクを登録
    # --- ▲ 修正 ▲ ---
}

__all__ = [
    "ANNBaselineModel", 
    "SST2Task", "CIFAR10Task", "MRPCTask", 
    # --- ▼ 修正 ▼ ---
    "CIFAR10DVSTask", 
    "SHDTask", # 公開リストに追加
    # --- ▲ 修正 ▲ ---
    "calculate_accuracy", "TASK_REGISTRY", "BenchmarkTask"
]