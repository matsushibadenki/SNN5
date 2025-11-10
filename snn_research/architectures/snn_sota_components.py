# ファイルパス: snn_research/architectures/snn_sota_components.py
# (v_fix_mypy_operator 修正 v2)
# Title: SOTA SNN コンポーネント (SNN-RMSNorm, SNN-SwiGLU, SNN-RoPE)
#
# Description:
# doc/ROADMAP.md (P1.2) および doc/レポート：ANN-SNN変換情報.md (IV) に基づき、
# SOTA Transformer (Llama 4等) で使用される主要コンポーネントの
# SNN実装（または近似実装）を提供します。
#
# 実装コンポーネント:
# - SNNRMSNorm (BrainTransformers [33] ベース)
# - SNNSwiGLU (BrainTransformers [33] ベース)
# - ConditionalPositionalEncoding (CPG-PE [2] / Spikformer V2 [35] ベースの近似SNN-RoPE)
#
# 修正 (v_fix_mypy_operator): [operator] "Tensor" not callable エラーを修正。
#                            (set_stateful, reset メソッド内の型推論エラーを cast で修正)
#
# 改善 (v3):
# - P1.2 のタスクに基づき、SNNSwiGLU と ConditionalPositionalEncoding (SNN-RoPE近似) を追加。
# - mypy --strict 準拠。

import torch
import torch.nn as nn
# --- ▼ 修正: math, F をインポート ▼ ---
from typing import Tuple, Dict, Any, Type, Optional, List, cast
import math
import torch.nn.functional as F
# --- ▲ 修正 ▲ ---

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


# --- ▼▼▼ P1.2 SNN-SwiGLU の実装 ▼▼▼ ---

class SNNSiLUApproximator(PiecewiseLinearApproximator):
    """
    SNN-SwiGLU用: SiLU(x) = x * sigmoid(x) 関数を近似する。
    BrainTransformers [33] に基づく。
    """
    def __init__(
        self,
        features: int,
        num_segments: int = 16, # SiLUは複雑なためセグメントを増やす
        neuron_class: Type[AdaptiveLIFNeuron] = AdaptiveLIFNeuron,
        neuron_params: Optional[Dict[str, Any]] = None
    ) -> None:
        if neuron_params is None:
            neuron_params = {'tau_mem': 10.0, 'base_threshold': 1.0}
        super().__init__(features, num_segments, neuron_class, neuron_params)
        
        # ターゲット関数 SiLU(x) = x * sigmoid(x) に基づいて
        # 重みを事前初期化 (オプションだが有効)
        with torch.no_grad():
            # (0, 1) の範囲で x をサンプリング
            x_sample = torch.linspace(0.01, 1.0, num_segments)
            y_silu = x_sample * torch.sigmoid(x_sample)
            # 簡易的な重み初期化 (各セグメントがその値を持つように)
            weights_data = y_silu - torch.cat([torch.tensor([0.0]), y_silu[:-1]])
            self.weights.data = weights_data.unsqueeze(0).repeat(features, 1)

        logger.info(f"SNN-SwiGLU: SNNSiLUApproximator (SiLU) initialized with {num_segments} segments.")

class SNNSwiGLU(sj_base.MemoryModule):
    """
    SNN-SwiGLU (ロードマップ P1.2)。
    BrainTransformers [33] に基づく。
    SwiGLU(x, W, V) = SiLU(xW) ⊗ xV
    
    SNN実装:
    gate = SNNSiLU(Linear_W(x))
    value = LIF(Linear_V(x))
    output = gate * value (アダマール積 / ANDゲート)
    
    注: この実装は時間ステップ (T) でのループを必要とします。
    """
    gate_proj: nn.Linear
    gate_silu: SNNSiLUApproximator
    value_proj: nn.Linear
    value_lif: AdaptiveLIFNeuron
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int, # SwiGLUの中間次元
        neuron_config: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        if neuron_type_str != 'lif':
            logger.warning("SNNSwiGLU defaulting to AdaptiveLIFNeuron.")
        neuron_class: Type[AdaptiveLIFNeuron] = AdaptiveLIFNeuron
        
        filtered_params: Dict[str, Any] = {
            k: v for k, v in neuron_params.items() 
            if k in ['tau_mem', 'base_threshold']
        }
        
        # 1. ゲートパス (xW)
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        # 2. SiLU近似 (SNN)
        self.gate_silu = SNNSiLUApproximator(
            features=d_ffn, neuron_class=neuron_class, neuron_params=filtered_params
        )
        
        # 3. バリューパス (xV)
        self.value_proj = nn.Linear(d_model, d_ffn, bias=False)
        # 4. バリューパスのLIF (標準的なLIF)
        self.value_lif = neuron_class(features=d_ffn, **filtered_params)

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        self.gate_silu.set_stateful(stateful)
        self.value_lif.set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        self.gate_silu.reset()
        self.value_lif.reset()

    def forward(self, x_spikes_seq: torch.Tensor) -> torch.Tensor:
        """
        スパイク時系列 (B, T, D_model) を SwiGLU処理 (スパイク) する。
        """
        B, T_seq, D = x_spikes_seq.shape
        
        SJ_F.reset_net(self)
        self.set_stateful(True)
        
        outputs: List[torch.Tensor] = []

        for t in range(T_seq):
            x_t: torch.Tensor = x_spikes_seq[:, t, :] # (B, D_model)
            
            # 1. ゲートパス (xW -> SiLU)
            gate_current: torch.Tensor = self.gate_proj(x_t)
            gate_spikes_t: torch.Tensor = self.gate_silu(gate_current) # (B, D_ffn)
            
            # 2. バリューパス (xV -> LIF)
            value_current: torch.Tensor = self.value_proj(x_t)
            value_spikes_t, _ = self.value_lif(value_current) # (B, D_ffn)
            
            # 3. 要素積 (⊗) -> スパイクドメインでは AND (アダマール積)
            y_t: torch.Tensor = gate_spikes_t * value_spikes_t
            
            outputs.append(y_t)
            
        self.set_stateful(False)
        
        return torch.stack(outputs, dim=1) # (B, T_seq, D_ffn)

# --- ▲▲▲ P1.2 SNN-SwiGLU の実装 ▲▲▲ ---


# --- ▼▼▼ P1.2 SNN-RoPE (近似) の実装 ▼▼▼ ---

class ConditionalPositionalEncoding(sj_base.MemoryModule):
    """
    近似 SNN-RoPE (ロードマップ P1.2)。
    Spikformer V2 [35] や CPG-PE [2] のアイデアに基づく、
    学習可能な「条件付き位置エンコーディング生成器」。
    
    RoPEの回転行列を近似する代わりに、位置インデックス(t)を入力とし、
    それに対応するスパイクベースの位置エンコーディング(PE_t)を生成する
    小規模なSNN (CPG) を実装する。
    """
    cpg_rnn: nn.RNN
    cpg_lif: AdaptiveLIFNeuron
    cpg_proj: nn.Linear
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        neuron_config: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 位置インデックス (0 ... max_seq_len-1) を受け取る埋め込み
        self.pos_idx_embed = nn.Embedding(max_seq_len, d_model)
        
        # CPG (中枢パターン生成器) を模倣する小規模RNN
        self.cpg_rnn = nn.RNN(d_model, d_model, batch_first=True)
        
        # CPGの出力をスパイク化するニューロン
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        if neuron_type_str != 'lif':
            logger.warning("SNNRoPE defaulting to AdaptiveLIFNeuron.")
        neuron_class: Type[AdaptiveLIFNeuron] = AdaptiveLIFNeuron
        filtered_params: Dict[str, Any] = {
            k: v for k, v in neuron_params.items() 
            if k in ['tau_mem', 'base_threshold']
        }
        self.cpg_lif = neuron_class(features=d_model, **filtered_params)
        
        logger.info(f"SNN-RoPE (CPG-PE Approx) [P1.2] initialized. MaxLen: {max_seq_len}")

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        self.cpg_lif.set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        self.cpg_lif.reset()

    def forward(
        self,
        x_spikes_seq: torch.Tensor # (B, T_seq, D)
    ) -> torch.Tensor:
        """
        入力スパイク時系列 (B, T, D) を受け取り、
        それに対応するスパイクベースの位置エンコーディングを加算する。
        """
        B, T_seq, D = x_spikes_seq.shape
        device = x_spikes_seq.device
        
        if T_seq > self.max_seq_len:
            logger.warning(f"SNN-RoPE: Input seq_len ({T_seq}) > max_seq_len ({self.max_seq_len}). Truncating.")
            T_seq = self.max_seq_len
            x_spikes_seq = x_spikes_seq[:, :T_seq, :]
            
        # 1. 位置インデックス (0, 1, ..., T_seq-1) を生成
        pos_indices = torch.arange(T_seq, device=device).unsqueeze(0).expand(B, -1) # (B, T_seq)
        
        # 2. CPG-RNNで位置エンコーディング（アナログ）を生成
        pos_emb_analog = self.pos_idx_embed(pos_indices) # (B, T_seq, D)
        pos_rnn_out, _ = self.cpg_rnn(pos_emb_analog) # (B, T_seq, D)
        
        # 3. スパイク化 (時間ステップ T_seq でループ)
        SJ_F.reset_net(self.cpg_lif)
        self.cpg_lif.set_stateful(True)
        
        pe_spikes_list: List[torch.Tensor] = []
        for t in range(T_seq):
            current_analog = pos_rnn_out[:, t, :] # (B, D)
            pe_spike_t, _ = self.cpg_lif(current_analog)
            pe_spikes_list.append(pe_spike_t)
            
        self.cpg_lif.set_stateful(False)
        
        pe_spikes_seq = torch.stack(pe_spikes_list, dim=1) # (B, T_seq, D)
        
        # 4. 入力スパイクと位置エンコーディング・スパイクを加算 (ORゲート)
        # (x > 0) | (pe > 0)
        output_spikes = (x_spikes_seq + pe_spikes_seq).clamp(0, 1)
        
        return output_spikes

# --- ▲▲▲ P1.2 SNN-RoPE (近似) の実装 ▲▲▲ ---