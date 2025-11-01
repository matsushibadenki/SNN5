# snn_research/learning_rules/stdp.py
# (修正)
# 修正: 階層的因果学習のため、戻り値の型を基底クラスに合わせる。

import torch
from typing import Dict, Any, Optional, Tuple
from .base_rule import BioLearningRule

class STDP(BioLearningRule):
    """ペアベースのSTDP学習ルールを実装するクラス。"""
    # ... (init, _initialize_traces, _update_traces は変更なし) ...
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, dt: float = 1.0):
        self.learning_rate = learning_rate
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_trace = tau_trace
        self.dt = dt
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None

    def _initialize_traces(self, pre_shape: int, post_shape: int, device: torch.device):
        """スパイクトレースを初期化する。"""
        self.pre_trace = torch.zeros(pre_shape, device=device)
        self.post_trace = torch.zeros(post_shape, device=device)
        
    def _update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """スパイクトレースを更新する。"""
        assert self.pre_trace is not None and self.post_trace is not None, "Traces not initialized."
            
        self.pre_trace = self.pre_trace - (self.pre_trace / self.tau_trace) * self.dt + pre_spikes
        self.post_trace = self.post_trace - (self.post_trace / self.tau_trace) * self.dt + post_spikes

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STDPに基づいて重み変化量を計算する。"""
        if self.pre_trace is None or self.post_trace is None or self.pre_trace.shape[0] != pre_spikes.shape[0] or self.post_trace.shape[0] != post_spikes.shape[0]:
            self._initialize_traces(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)

        self._update_traces(pre_spikes, post_spikes)

        assert self.pre_trace is not None and self.post_trace is not None

        dw = torch.zeros_like(weights)
        dw += self.learning_rate * self.a_plus * torch.outer(post_spikes, self.pre_trace)
        dw -= self.learning_rate * self.a_minus * torch.outer(pre_spikes, self.post_trace).T
        
        return dw, None