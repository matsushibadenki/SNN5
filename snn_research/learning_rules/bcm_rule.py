# ファイルパス: snn_research/learning_rules/bcm_rule.py
# (新規作成)
#
# Title: BCM (Bienenstock-Cooper-Munro) 学習規則
#
# Description:
# doc/プロジェクト強化案の調査.md (セクション2.3, 引用[35, 38]) に基づく実装。
# ヘブ則（STDPなど）による学習の不安定性を解消するための、
# 生物学的に妥当なホメオスタシス（恒常性維持）メカニズム。
#
# シナプス後ニューロンの長期的な平均活動 (avg_post_activity) に基づいて
# 可塑性の閾値 (theta) を動的に変更し、発火率を安定させる。
#
# mypy --strict 準拠。

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class BCMLearningRule(BioLearningRule):
    """
    BCM (Bienenstock-Cooper-Munro) 学習規則。
    ニューロンの平均活動に基づいて可塑性閾値を動的に調整し、
    ネットワークの恒常性を維持する。
    """
    # (B, N_post) の長期的な平均活動を保持するバッファ
    avg_post_activity: Optional[torch.Tensor]

    def __init__(
        self, 
        learning_rate: float, 
        tau_avg: float, # 平均活動を計算するための時定数 (ステップ数)
        target_rate: float, # 目標とする平均発火率 (0-1)
        dt: float = 1.0
    ):
        """
        Args:
            learning_rate (float): 学習率。
            tau_avg (float): 平均活動の時定数。大きいほど長期的な平均になる。
            target_rate (float): 目標とする平均発火率 (例: 0.1)。
            dt (float): 時間ステップ。
        """
        self.learning_rate = learning_rate
        if tau_avg <= 0:
            raise ValueError("tau_avg must be positive")
        self.tau_avg = tau_avg
        if not (0 < target_rate <= 1.0):
             raise ValueError("target_rate must be between 0 and 1.0")
        self.target_rate = target_rate
        self.dt = dt
        
        self.avg_post_activity = None
        
        # 指数移動平均の係数 (alpha = dt / tau)
        self.avg_decay_factor = dt / self.tau_avg

        print(f"🧠 BCM Learning Rule initialized (Target Rate: {target_rate}, Tau Avg: {tau_avg})")

    def _initialize_traces(self, post_shape: int, device: torch.device):
        """平均活動トレースを初期化する。"""
        # (N_post,) の形状で初期化 (バッチ非依存の長期平均)
        # BCMはニューロン単位のホメオスタシス
        self.avg_post_activity = torch.full((post_shape,), self.target_rate, device=device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        BCM則に基づいて重み変化量を計算する。
        dw = lr * pre_spikes * phi(post_spikes, avg_post_activity)
        
        Args:
            pre_spikes (torch.Tensor): (N_pre,) または (B, N_pre)
            post_spikes (torch.Tensor): (N_post,) または (B, N_post)
            weights (torch.Tensor): (N_post, N_pre)
        """
        # バッチ処理に対応 (B, N) -> (N,)
        if pre_spikes.dim() > 1:
            pre_spikes_avg = pre_spikes.mean(dim=0)
        else:
            pre_spikes_avg = pre_spikes
            
        if post_spikes.dim() > 1:
            post_spikes_avg = post_spikes.mean(dim=0)
        else:
            post_spikes_avg = post_spikes

        # --- 1. トレースの初期化 ---
        if self.avg_post_activity is None or self.avg_post_activity.shape[0] != post_spikes_avg.shape[0]:
            self._initialize_traces(post_spikes_avg.shape[0], pre_spikes.device)
        
        avg_post_activity = cast(torch.Tensor, self.avg_post_activity)

        # --- 2. 長期平均活動の更新 (指数移動平均) ---
        # avg[t] = (1 - alpha) * avg[t-1] + alpha * post_spikes
        with torch.no_grad():
            self.avg_post_activity = (
                (1.0 - self.avg_decay_factor) * avg_post_activity + 
                self.avg_decay_factor * post_spikes_avg
            ).detach() # 勾配計算には不要

        # --- 3. BCM閾値 (theta) の計算 ---
        # theta = E[post]^2 / target_rate (引用[38]に基づく単純化)
        # または theta = E[post] (より一般的)
        # ここでは theta = E[post] (現在の平均活動) を使用
        theta = avg_post_activity.clone()
        
        # --- 4. BCM関数 (phi) の計算 ---
        # phi = post * (post - theta)
        # LTP (post > theta) と LTD (post < theta) を引き起こす
        
        # (N_post,)
        phi = post_spikes_avg * (post_spikes_avg - theta)
        
        # --- 5. 重み変化量 (dw) の計算 ---
        # dw = lr * phi * pre_spikes^T
        # (N_post,) * (N_pre,) -> (N_post, N_pre)
        dw = self.learning_rate * torch.outer(phi, pre_spikes_avg)
        
        # 安定化のための重み減衰 (オプション)
        # dw -= self.learning_rate * 0.001 * weights

        # BCMは局所的なルールであり、逆方向のクレジット信号は生成しない
        backward_credit = None

        return dw, backward_credit