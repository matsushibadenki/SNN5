# ファイルパス: snn_research/architectures/snn_sota_components.py
# (v_fix_mypy_operator 修正 v2)
# Title: SOTA SNN コンポーネント (SNN-RMSNorm)
#
# Description:
# ... (中略) ...
# 修正 (v_fix_mypy_operator): [operator] "Tensor" not callable エラーを修正。
#                            (set_stateful, reset メソッド内の型推論エラーを cast で修正)

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Type, Optional, List, cast

# SNNのコアコンポーネントをインポート
from snn_research.core.base import SNNLayerNorm # SNNLayerNormは標準LNのため、ここではRMSNormをカスタム実装
from snn_research.core.neurons import AdaptiveLIFNeuron
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]
from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]

import logging

logger = logging.getLogger(__name__)

# --- SNN-RMSNorm サブモジュール (BrainTransformers [33] ベース) ---

class PiecewiseLinearApproximator(sj_base.MemoryModule):
    """
    区分的線形近似（Piecewise Linear Approximation）を用いて
    任意の非線形関数 (f(x)) を近似する、汎用SNNモジュール。
    
    LAS [36] の HG ニューロンや BrainTransformers [33] の
    Square/Sqrt Approximator の基礎となるアイデア。
    
    ここでは、複数のLIFニューロンが異なる閾値と重みを持ち、
    入力空間の異なる領域を担当することをシミュレートします。
    """
    lif_neurons: nn.ModuleList
    weights: nn.Parameter

    def __init__(
        self,
        features: int,
        num_segments: int, # 近似に使用する区分（ニューロン）の数
        neuron_class: Type[AdaptiveLIFNeuron],
        neuron_params: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.features = features
        self.num_segments = num_segments
        
        self.lif_neurons = nn.ModuleList()
        
        # 各区分 (セグメント) ごとに異なる閾値を持つLIFニューロンを生成
        base_threshold = neuron_params.get('base_threshold', 1.0)
        
        for i in range(num_segments):
            seg_params = neuron_params.copy()
            # 閾値を線形に増加させる (0.1, 0.2, ..., 1.0)
            seg_params['base_threshold'] = base_threshold * (i + 1) / num_segments
            self.lif_neurons.append(neuron_class(features=features, **seg_params))
            
        # 各セグメント（ニューロン）の出力をスケーリングする学習可能な重み
        self.weights = nn.Parameter(torch.randn(features, num_segments) * (1.0 / num_segments))

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        # --- ▼ 修正: [operator] mypy型推論エラーを cast で修正 ▼ ---
        for lif_module in self.lif_neurons:
            lif = cast(AdaptiveLIFNeuron, lif_module)
            lif.set_stateful(stateful)
        # --- ▲ 修正 ▲ ---

    def reset(self) -> None:
        super().reset()
        # --- ▼ 修正: [operator] mypy型推論エラーを cast で修正 ▼ ---
        for lif_module in self.lif_neurons:
            lif = cast(AdaptiveLIFNeuron, lif_module)
            lif.reset()
        # --- ▲ 修正 ▲ ---

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        1タイムステップ分の処理。
        Args:
            x_t (torch.Tensor): 入力 (B, Features)
        Returns:
            torch.Tensor: 近似された関数 f(x_t) の出力 (B, Features)
        """
        B, F = x_t.shape
        
        segment_spikes: List[torch.Tensor] = []
        
        # 1. 各セグメントのニューロンが、それぞれの閾値で発火
        for lif_module in self.lif_neurons:
            lif = cast(AdaptiveLIFNeuron, lif_module)
            spike_seg, _ = lif(x_t) # (B, F)
            segment_spikes.append(spike_seg)
            
        # (K, B, F) -> (B, F, K)
        all_segment_spikes: torch.Tensor = torch.stack(segment_spikes, dim=0).permute(1, 2, 0)
        
        # 2. 学習可能な重みでスケーリングして合計
        # (B, F, K) * (F, K) -> (B, F)
        y_t: torch.Tensor = (all_segment_spikes * self.weights).sum(dim=-1)
        
        return y_t

class SquareApproximator(PiecewiseLinearApproximator):
    """
    SNN-RMSNorm用: x^2 関数を近似する。
    """
    def __init__(
        self,
        features: int,
        num_segments: int = 8,
        neuron_class: Type[AdaptiveLIFNeuron] = AdaptiveLIFNeuron,
        neuron_params: Optional[Dict[str, Any]] = None
    ) -> None:
        if neuron_params is None:
            neuron_params = {'tau_mem': 10.0, 'base_threshold': 1.0}
        super().__init__(features, num_segments, neuron_class, neuron_params)
        logger.info(f"SNN-RMSNorm: SquareApproximator (x^2) initialized with {num_segments} segments.")

class SqrtApproximator(PiecewiseLinearApproximator):
    """
    SNN-RMSNorm用: 1/sqrt(x) 関数を近似する。
    (RMSNormは x * (1/sqrt(mean(x^2))) の計算のため)
    """
    def __init__(
        self,
        features: int,
        num_segments: int = 8,
        neuron_class: Type[AdaptiveLIFNeuron] = AdaptiveLIFNeuron,
        neuron_params: Optional[Dict[str, Any]] = None
    ) -> None:
        if neuron_params is None:
            neuron_params = {'tau_mem': 10.0, 'base_threshold': 1.0}
        super().__init__(features, num_segments, neuron_class, neuron_params)
        logger.info(f"SNN-RMSNorm: SqrtApproximator (1/sqrt(x)) initialized with {num_segments} segments.")

class SNNRMSNorm(sj_base.MemoryModule):
    """
    SNN-RMSNorm (ロードマップ P1.2)。
    BrainTransformers [33] のアイデアに基づくスタブ実装。
    
    注: この実装は時間ステップ (T) でのループを必要とします。
    """
    square_approx: SquareApproximator
    sqrt_approx: SqrtApproximator
    
    # 時間積分のためのLIFニューロン
    mean_integrator: AdaptiveLIFNeuron
    
    def __init__(
        self,
        d_model: int,
        time_steps: int,
        neuron_config: Dict[str, Any],
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.eps = eps
        
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        if neuron_type_str != 'lif':
            logger.warning("SNNRMSNorm defaulting to AdaptiveLIFNeuron.")
        neuron_class: Type[AdaptiveLIFNeuron] = AdaptiveLIFNeuron
        
        filtered_params: Dict[str, Any] = {
            k: v for k, v in neuron_params.items() 
            if k in ['tau_mem', 'base_threshold']
        }

        # 1. x^2 を近似するモジュール
        self.square_approx = SquareApproximator(d_model, neuron_class=neuron_class, neuron_params=filtered_params)
        
        # 2. 平均を時間積分するニューロン (tau=T_seq とすると平均化に近い)
        mean_params = filtered_params.copy()
        mean_params['tau_mem'] = float(time_steps) # 時定数をTに設定
        self.mean_integrator = neuron_class(features=d_model, **mean_params)
        
        # 3. 1/sqrt(x) を近似するモジュール
        self.sqrt_approx = SqrtApproximator(d_model, neuron_class=neuron_class, neuron_params=filtered_params)

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        self.square_approx.set_stateful(stateful)
        self.mean_integrator.set_stateful(stateful)
        self.sqrt_approx.set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        self.square_approx.reset()
        self.mean_integrator.reset()
        self.sqrt_approx.reset()

    def forward(self, x_spikes_seq: torch.Tensor) -> torch.Tensor:
        """
        スパイク時系列 (B, T, D_model) を RMSNorm する。
        
        Args:
            x_spikes_seq (torch.Tensor): 入力スパイク時系列 (B, T_seq, D_model)
        
        Returns:
            torch.Tensor: 正規化されたスパイク時系列 (B, T_seq, D_model)
        """
        B, T_seq, D = x_spikes_seq.shape
        
        SJ_F.reset_net(self)
        self.set_stateful(True)
        
        # --- 時間ステップ (T_seq) でループ ---
        
        # (B, D)
        h_mean_state: torch.Tensor = torch.zeros(B, D, device=x_spikes_seq.device)
        
        outputs: List[torch.Tensor] = []

        for t in range(T_seq):
            x_t: torch.Tensor = x_spikes_seq[:, t, :] # (B, D)
            
            # 1. x_t^2 をスパイクで近似
            # x_t (スパイク) -> x_t^2 (アナログ値)
            # SNN-RMSNorm [33] は入力がアナログ値(膜電位)を想定している可能性が高い
            # ここでは入力 x_t (スパイク) をそのまま使う (簡易実装)
            x_t_squared_spikes: torch.Tensor = self.square_approx(x_t) # (B, D)
            
            # 2. mean(x_t^2) を時間積分で近似
            # h_mean[t] = LIF(x_t^2) (tau=T_seq)
            _, h_mean_state = self.mean_integrator(x_t_squared_spikes) # (B, D)
            
            # 3. 1 / sqrt(mean + eps) を近似
            # (h_mean_state は膜電位(アナログ値)であり、SqrtApproximatorの入力として適切)
            inv_rms_spikes: torch.Tensor = self.sqrt_approx(h_mean_state + self.eps) # (B, D)
            
            # 4. y_t = x_t * (1 / sqrt(mean + eps))
            # (スパイク * スパイク) -> ANDゲート (アダマール積)
            y_t: torch.Tensor = x_t * inv_rms_spikes
            
            outputs.append(y_t)

        self.set_stateful(False)
        
        return torch.stack(outputs, dim=1) # (B, T_seq, D)
