# ファイルパス: snn_research/training/losses.py
# Title: 損失関数定義 (Distillation, Regularization)
#
# 機能の説明: 知識蒸留 (KD)、スパイク正則化、スパース性正則化など、
# SNNの学習に使用するカスタム損失関数を定義する。
#
# 【修正内容 v31.2: 循環インポート (Circular Import) の修正】
# - health-check (項目2) の 'ImportError: ... (benchmark.tasks)'
#   の根本原因となっていた循環参照を修正します。
# - (L: 21) 'from snn_research.core.snn_core import SNNCore' が、
#   snn_core.py (L:29) -> benchmark.tasks (L:38) -> losses.py (L:21)
#   という循環参照を引き起こしていました。
# - (L: 24, 76, 122) 'BaseLoss(SNNCore)' や 'DistillationLoss(BaseLoss)'
#   が 'SNNCore' を継承するのは誤りです。
# - 損失関数は 'nn.Module' を継承すべきです。
# - (L: 21) 'SNNCore' のインポートを削除しました。
# - (L: 24, 76, 122) 継承元を 'nn.Module' に変更しました。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 !!!】 ▼▼▼
# (from snn_research.core.snn_core import SNNCore を削除)
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲


# === 1. 基底損失クラス (v17) ===

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 !!!】 ▼▼▼
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

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 !!!】 ▼▼▼
class DistillationLoss(BaseLoss): # BaseLoss (nn.Module) を継承
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲
    """
    (v17)
    知識蒸留 (Knowledge Distillation) 損失。
    通常の CrossEntropy (CE) 損失と、
    教師モデルのロジットとの Kullback-Leibler (KL) ダイバージェンス損失を
    重み付けして合計する。
    
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

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 !!!】 ▼▼▼
class SpikeRegularizationLoss(BaseLoss): # BaseLoss (nn.Module) を継承
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲
    """
    (v17)
    スパイク発火率 (平均スパイク数) に対する L2 正則化損失。
    スパイク活動を抑制（または促進）するために使用する。
    
    (HPO (Turn 5) (L:341) の 'spike_reg_loss' を計算)
    """
    def __init__(self, weight: float = 1e-4, target_rate: float = 0.0):
        super().__init__(weight=weight)
        self.target_rate = float(target_rate)
        
    def forward(self, avg_spikes: torch.Tensor) -> torch.Tensor:
        """
        (v17)
        Args:
            avg_spikes (torch.Tensor): モデル (またはレイヤー) の
                                       平均スパイク数 (スカラー)

        Returns:
            torch.Tensor: L2 損失 (スカラー)
        """
        # (v17) (平均スパイク - ターゲット)^2
        loss = (avg_spikes - self.target_rate) ** 2
        return loss


# === 4. スパース性正則化損失 (v17) ===

class SparsityRegularizationLoss(BaseLoss):
    """
    (v17)
    スパイク発火率 (平均スパイク数) に対する L1 正則化損失。
    スパース性 (発火数を 0 に近づける) を促進するために使用する。
    
    (HPO (Turn 5) (L:341) の 'sparsity_loss' を計算)
    """
    def __init__(self, weight: float = 1e-4):
        super().__init__(weight=weight)
        
    def forward(self, avg_spikes: torch.Tensor) -> torch.Tensor:
        """
        (v17)
        Args:
            avg_spikes (torch.Tensor): モデル (またはレイヤー) の
                                       平均スパイク数 (スカラー)

        Returns:
            torch.Tensor: L1 損失 (スカラー)
        """
        # (v17) |平均スパイク| (L1)
        loss = torch.abs(avg_spikes)
        return loss
