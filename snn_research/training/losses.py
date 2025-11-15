# ファイルパス: snn_research/training/losses.py
# Title: 損失関数定義 (Distillation, Regularization)
#
# 機能の説明: 知識蒸留 (KD)、スパイク正則化、スパース性正則化など、
# SNNの学習に使用するカスタム損失関数を定義する。
#
# 【修正内容 v32: ImportError (cannot import name 'CombinedLoss') の修正】
# - health-check 実行時に 'ImportError: cannot import name 'CombinedLoss''
#   が発生する問題に対処します。
# - v31.2 の循環インポート修正時に、'CombinedLoss' クラス
#   がファイルから欠落していました。
# - (L: 186) 'CombinedLoss'
#   の定義をファイル末尾に追加しました。
# - (v31.2の修正も維持) 'BaseLoss' (L: 24)
#   および 'CombinedLoss' (L: 186)
#   が 'SNNCore' ではなく
#   'nn.Module' を継承するようにし、循環インポートを回避します。
#
# 【修正内容 v31.2: 循環インポート (Circular Import) の修正】
# - (v32でも維持) 'SNNCore'
#   へのインポート (L: 21) を削除。
# - (v32でも維持) 継承元を 'nn.Module' に変更 (L: 24, 76, 122)。

import torch
import torch.nn as nn
import torch.nn.functional as F
# (v32: 'CombinedLoss' の 'List' のために追加)
from typing import Dict, Any, Optional, Tuple, List, Union

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 (v32でも維持) !!!】 ▼▼▼
# (from snn_research.core.snn_core import SNNCore を削除)
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲


# === 1. 基底損失クラス (v17) ===

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 (v32でも維持) !!!】 ▼▼▼
class BaseLoss(nn.Module): # 'SNNCore' -> 'nn.Module' に変更
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲
    """
    (v17)
    カスタム損失関数の基底クラス。
    重み (weight) の管理と、'forward' のインターフェースを定義する。
    
    (v31.2) SNNCore ではなく nn.Module を継承する。
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        # (v17) 損失の重みを float として登録
        self.weight = float(weight)
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        (v17)
        損失を計算して返す (スカラーのテンソル)。
        """
        raise NotImplementedError
        
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        (v17)
        損失を計算し、重みを乗算して返す。
        """
        if self.weight == 0:
            return torch.tensor(0.0, 
                                device=self._get_device_from_args(*args, **kwargs),
                                dtype=torch.float32)
            
        loss = super().__call__(*args, **kwargs)
        return loss * self.weight

    def _get_device_from_args(self, *args, **kwargs) -> torch.device:
        """ (v17) 引数からデバイスを推測する """
        if args and isinstance(args[0], torch.Tensor):
            return args[0].device
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device('cpu')


# === 2. 知識蒸留 (KD) 損失 (v17) ===

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 (v32でも維持) !!!】 ▼▼▼
class DistillationLoss(BaseLoss): # BaseLoss (nn.Module) を継承
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲
    """
    (v17)
    知識蒸留 (Knowledge Distillation) 損失。
    (HPO (Turn 5) (L:341) の 'distill_loss' (KD) と 
     'ce_loss' を計算するために使用)
    """
    def __init__(
        self,
        ce_weight: float = 0.5,
        distill_weight: float = 0.5,
        temperature: float = 2.0,
        **kwargs # (v17: BaseLoss の weight を吸収)
    ):
        # (v17) BaseLoss の weight は 1.0 (固定)
        super().__init__(weight=1.0) 
        
        self.ce_weight = float(ce_weight)
        self.distill_weight = float(distill_weight)
        self.temperature = float(temperature)
        
        # (v17) KLダイバージェンス損失 (KD用)
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        (v17)
        Args:
            student_logits: 生徒モデルの出力 (B, NumClasses)
            teacher_logits: 教師モデルの出力 (B, NumClasses)
            targets: 正解ラベル (B,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (total_loss, ce_loss, kd_loss)
        """
        
        # 1. CrossEntropy (CE) 損失
        ce_loss = F.cross_entropy(student_logits, targets)
        
        # 2. Kullback-Leibler (KL) 損失 (KD)
        kd_loss = self.kl_div_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2) # (T^2 スケーリング)
        
        # 3. 総損失 (重み付け)
        total_loss = (self.ce_weight * ce_loss) + \
                     (self.distill_weight * kd_loss)
                     
        return total_loss, ce_loss, kd_loss


# === 3. スパイク正則化損失 (v17) ===

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 (v32でも維持) !!!】 ▼▼▼
class SpikeRegularizationLoss(BaseLoss): # BaseLoss (nn.Module) を継承
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲
    """
    (v17)
    スパイク発火率 (平均スパイク数) に対する L2 正則化損失。
    (HPO (Turn 5) (L:341) の 'spike_reg_loss' を計算)
    """
    def __init__(self, weight: float = 1e-4, target_rate: float = 0.0):
        super().__init__(weight=weight)
        self.target_rate = float(target_rate)
        
    def forward(self, avg_spikes: torch.Tensor) -> torch.Tensor:
        """
        (v17)
        Args:
            avg_spikes (torch.Tensor): 平均スパイク数 (スカラー)
        Returns:
            torch.Tensor: L2 損失 (スカラー)
        """
        loss = (avg_spikes - self.target_rate) ** 2
        return loss


# === 4. スパース性正則化損失 (v17) ===

class SparsityRegularizationLoss(BaseLoss):
    """
    (v17)
    スパイク発火率 (平均スパイク数) に対する L1 正則化損失。
    (HPO (Turn 5) (L:341) の 'sparsity_loss' を計算)
    """
    def __init__(self, weight: float = 1e-4):
        super().__init__(weight=weight)
        
    def forward(self, avg_spikes: torch.Tensor) -> torch.Tensor:
        """
        (v17)
        Args:
            avg_spikes (torch.Tensor): 平均スパイク数 (スカラー)
        Returns:
            torch.Tensor: L1 損失 (スカラー)
        """
        loss = torch.abs(avg_spikes)
        return loss


# --- ▼▼▼ 【!!! 修正 v32: 欠落していた 'CombinedLoss' を追加 !!!】 ▼▼▼
class CombinedLoss(BaseLoss):
    """
    (v17)
    複数の損失関数 (BaseLoss のサブクラス) を
    辞書またはリストで受け取り、合計するラッパー。
    
    (v31.2 / v32) BaseLoss (nn.Module) を継承
    """
    def __init__(
        self, 
        loss_functions: Union[Dict[str, BaseLoss], List[BaseLoss]],
        **kwargs # (v17: BaseLoss の weight を吸収)
    ):
        super().__init__(weight=1.0) # (v17: ラッパー自体は重み 1.0)
        
        if isinstance(loss_functions, dict):
            self.loss_functions = nn.ModuleDict(loss_functions)
        elif isinstance(loss_functions, list):
            # (v17) 辞書に変換 (ModuleList は辞書を返せないため)
            self.loss_functions = nn.ModuleDict(
                {f"loss_{i}": fn for i, fn in enumerate(loss_functions)}
            )
        else:
            raise TypeError(
                f"loss_functions must be a dict or list, "
                f"got {type(loss_functions)}"
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        (v17)
        全ての損失関数を実行し、合計 (スカラー) を返す。
        
        Returns:
            torch.Tensor: 合計損失 (スカラー)
        """
        total_loss = torch.tensor(0.0, 
                                  device=self._get_device_from_args(*args, **kwargs),
                                  dtype=torch.float32)
        
        # (v17) ModuleDict をイテレート
        for name, loss_fn in self.loss_functions.items():
            # (v17) BaseLoss.__call__ (重み付け) を実行
            loss = loss_fn(*args, **kwargs) 
            total_loss += loss
            
        return total_loss
# --- ▲▲▲ 【!!! 修正 v32】 ▲▲▲
