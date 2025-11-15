# ファイルパス: snn_research/benchmark/tasks.py
# Title: ベンチマークタスク定義
#
# 機能の説明: 分類 (CIFAR10) や回帰などの標準的なベンチマークタスクの
# 損失関数、メトリクス計算、入出力処理を定義する。
#
# 【修正内容 v30.2: 循環インポート (Circular Import) の修正】
# - health-check 実行時に 'ImportError: cannot import name 'get_task_by_name'
#   (most likely due to a circular import)' が発生する問題に対処します。
# - (L: 37) 'from snn_research.core.snn_core import SNNCore' が、
#   snn_core.py (L:29) -> tasks.py (L:37) という循環参照を引き起こしていました。
# - (L: 52) 'BaseTask(SNNCore)' という継承は誤りです。
#   タスクはモデル管理クラス (SNNCore) ではなく、
#   'nn.Module' を継承すべきです。
# - (L: 37, 52) 'SNNCore' への参照を削除し、'nn.Module' を継承するように
#   修正しました。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
import logging
from abc import ABC, abstractmethod

# (v17: snn_core.py (L:28) からの循環参照を避けるため、
#  SNNCore への参照を削除)
# --- ▼▼▼ 【!!! 修正 v30.2: 循環インポート修正 !!!】 ▼▼▼
# (from snn_research.core.snn_core import SNNCore を削除)
# --- ▲▲▲ 【!!! 修正 v30.2】 ▲▲▲

# (v17: 損失関数のインポート)
from ..training.losses import (
    DistillationLoss, 
    SpikeRegularizationLoss, 
    SparsityRegularizationLoss
)
# (v17: メトリクスのインポート)
from .metrics import accuracy

logger = logging.getLogger(__name__)

# === 基底タスク (v17) ===

# --- ▼▼▼ 【!!! 修正 v30.2: 循環インポート修正 !!!】 ▼▼▼
class BaseTask(nn.Module, ABC): # 'SNNCore' -> 'nn.Module, ABC' に変更
# --- ▲▲▲ 【!!! 修正 v30.2】 ▲▲▲
    """
    (v17)
    タスク定義の抽象基底クラス (ABC)。
    損失関数、メトリクス、データ処理をカプセル化する。
    
    (v30.2) SNNCore ではなく nn.Module を継承する。
    """
    def __init__(
        self,
        task_config: Dict[str, Any],
        data_config: Dict[str, Any],
        device: torch.device
    ):
        super().__init__()
        self.task_config = task_config
        self.data_config = data_config
        self.device = device
        
        # (v17) データから num_classes を取得
        self.num_classes = int(self.data_config.get("num_classes", 10))
        
        # (v17) 損失関数の初期化 (サブクラスで行う)
        self._init_loss_functions()

    @abstractmethod
    def _init_loss_functions(self):
        """ (v17) 損失関数 (例: CE, Distillation) を初期化する """
        raise NotImplementedError

    @abstractmethod
    def process_output(
        self, 
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
        targets: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        (v17)
        モデルの出力 (タプルの場合もある) とターゲットを受け取り、
        損失 (loss) とメトリクス (logits, accuracy など) の
        辞書を計算して返す。

        Args:
            model_output: モデルのforward()からの戻り値。
                          (logits) または (logits, avg_spikes, avg_mem)
            targets: 正解ラベル (B,)

        Returns:
            Dict[str, Any]: 損失とメトリクスを含む辞書
        """
        raise NotImplementedError
        
    def forward(self, *args, **kwargs):
        """ (v30.2) nn.Module のためのダミー forward """
        raise NotImplementedError(
            "BaseTask は直接呼び出されません。"
            "process_output() を使用してください。"
        )

# === 分類タスク (v17) ===

class CIFAR10Task(BaseTask):
    """
    (v17)
    CIFAR-10 分類タスク (Distillation Loss 対応)
    """
    def __init__(
        self,
        task_config: Dict[str, Any],
        data_config: Dict[str, Any],
        device: torch.device
    ):
        # (v17) 画像サイズ (img_size) の上書きを許可
        # (HPO (Turn 5) で 'Using custom img_size: 32' が
        #  表示されていたため)
        override_img_size = task_config.get("img_size")
        if override_img_size:
            logger.info(f"INFO (CIFAR10Task): Using custom img_size: {override_img_size}")
            data_config["img_size"] = override_img_size
            
        super().__init__(task_config, data_config, device)

    def _init_loss_functions(self):
        """
        (v17)
        HPO (Turn 5) のログに基づき、
        DistillationLoss, SpikeRegularizationLoss, SparsityRegularizationLoss
        を初期化する。
        """
        
        # (v17) HPO (Turn 5) のログ (L:33) から
        #       loss_config のパスを推定
        loss_config_path = "training.gradient_based.distillation.loss"
        loss_config = self.task_config.get_nested(loss_config_path, {})
        
        # (v17) 1. Distillation Loss (CE + KD)
        self.distill_loss_fn = DistillationLoss(
            ce_weight=loss_config.get("ce_weight", 0.5),
            distill_weight=loss_config.get("distill_weight", 0.5),
            temperature=loss_config.get("temperature", 2.0)
        )
        
        # (v17) 2. Spike Regularization
        self.spike_reg_fn = SpikeRegularizationLoss(
            weight=loss_config.get("spike_reg_weight", 1e-4)
        )
        
        # (v17) 3. Sparsity Regularization
        self.sparsity_reg_fn = SparsityRegularizationLoss(
            weight=loss_config.get("sparsity_reg_weight", 1e-4)
        )
        
        # (v17) 4. (HPO (Turn 5) のログ (L:341-493) には
        #       mem_reg_loss, temporal_compression_loss, 
        #       sparsity_threshold_reg_loss もあるが、
        #       重みが 0 または HPO パラメータにないため、ここでは省略)

    def process_output(
        self, 
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
        targets: Optional[torch.Tensor],
        teacher_logits: Optional[torch.Tensor] = None # (v17) KD用
    ) -> Dict[str, Any]:
        """
        (v17)
        HPO (Turn 5) のログ (L:341-493) に合わせて
        損失とメトリクスを計算する。
        
        Args:
            model_output: (logits, avg_spikes, avg_mem) のタプル
            targets: 正解ラベル (B,)
            teacher_logits: 教師モデルのロジット (B, NumClasses) (KD用)

        Returns:
            Dict[str, Any]: 損失とメトリクス
        """
        
        # 1. モデル出力のアンパック
        if not isinstance(model_output, (tuple, list)) or len(model_output) < 3:
            raise ValueError(
                f"CIFAR10Task は (logits, avg_spikes, avg_mem) の "
                f"タプルを期待しますが、受け取った型は {type(model_output)} です。"
            )
            
        logits, avg_spikes, avg_mem = model_output
        
        # (v17) HPO (Turn 5) (L:351) の 'avg_cutoff_steps' は
        #       4番目の戻り値かもしれないが、ここでは無視する
        
        metrics = {
            "logits": logits,
            "avg_spikes": avg_spikes,
            "avg_mem": avg_mem,
            # (v17) HPO (Turn 5) (L:351) 'avg_cutoff_steps=32'
            #       (これはトレーナー側で計算される可能性が高い)
        }

        # 2. 損失計算 (ターゲットと教師ロジットがある場合)
        if targets is not None and teacher_logits is not None:
            # (v17) HPO (Turn 5) (L:341) 
            #       'total=0.396, ce_loss=2.37, distill_loss=0.116, 
            #        spike_reg_loss=0.0004, sparsity_loss=2.86e-14'
            
            # (v17) 2a. Distillation Loss (CE + KD)
            distill_loss, ce_loss, kd_loss = self.distill_loss_fn(
                logits, teacher_logits, targets
            )
            metrics["ce_loss"] = ce_loss
            metrics["distill_loss"] = kd_loss # (ログ (L:341) の distill_loss は KD loss)
            
            # (v17) 2b. Spike Regularization
            spike_reg_loss = self.spike_reg_fn(avg_spikes)
            metrics["spike_reg_loss"] = spike_reg_loss
            
            # (v17) 2c. Sparsity Regularization
            # (注: ログ (L:341) では avg_spikes ではなく 
            #  'spike_rate=2.86e-14' を使っている可能性があるが、
            #  ここでは avg_spikes を代用)
            sparsity_loss = self.sparsity_reg_fn(avg_spikes)
            metrics["sparsity_loss"] = sparsity_loss

            # (v17) 2d. 総損失 (Total Loss)
            # (HPO (Turn 5) (L:341) の 'total' に合わせる)
            total_loss = distill_loss + spike_reg_loss + sparsity_loss
            metrics["loss"] = total_loss # (トレーナーが 'loss' を参照する)
            metrics["total"] = total_loss
            
            # (v17) 2e. メトリクス (Accuracy)
            # (HPO (Turn 5) (L:341) 'accuracy=0.188')
            acc = accuracy(logits, targets)
            metrics["accuracy"] = acc

        elif targets is not None:
            # (v17) 評価モード (教師ロジットなし)
            ce_loss = F.cross_entropy(logits, targets)
            metrics["loss"] = ce_loss
            metrics["ce_loss"] = ce_loss
            metrics["accuracy"] = accuracy(logits, targets)
            
            # (v17) 評価時も正則化損失を計算 (ログ用)
            metrics["spike_reg_loss"] = self.spike_reg_fn(avg_spikes)
            metrics["sparsity_loss"] = self.sparsity_reg_fn(avg_spikes)
            
        return metrics

# === タスクレジストリ (v17) ===

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    # (v17) HPO (Turn 5) (L:20) で 'cifar10' が指定
    "cifar10": CIFAR10Task,
    
    # (v17) 'train.py' (L:38) で
    # 'classification' がデフォルト
    "classification": CIFAR10Task, 
}

def get_task_by_name(name: str) -> Type[BaseTask]:
    """
    (v17)
    タスク名 (文字列) に基づいて、
    タスククラス (BaseTask) を返します。
    (snn_core.py L: 29 から呼び出される)
    """
    name_lower = name.lower()
    if name_lower not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task name: '{name}'. "
            f"Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name_lower]

# (v17) ユーティリティ (Dict.get_nested)
# (v17: BaseTask が SNNCore を継承しなくなったため、
#  BaseTask 自身 (またはこのモジュール) がヘルパーを持つ必要がある)
def _get_nested_config(config: Dict[str, Any], keys: str, default: Any = None) -> Any:
    """
    (v17)
    ネストした辞書から 'a.b.c' 形式のキーで値を取得する
    """
    keys_list = keys.split('.')
    current = config
    try:
        for key in keys_list:
            if isinstance(current, dict):
                current = current[key]
            else:
                # (v17) 'features' などのキーが途中で見つからない場合
                return default
        return current
    except (KeyError, TypeError):
        return default

# (v17) BaseTask にメソッドを追加
setattr(BaseTask, 'get_nested', _get_nested_config)
