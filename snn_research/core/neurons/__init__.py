# ファイルパス: snn_research/core/neurons/__init__.py
# (更新)
# Title: SNNニューロンモデル定義
#
# 【最終修正 v_fix_bias_logic】:
# - AdaptiveLIFNeuron にバイアス (bias_init) を受け取る機能を追加。
# - forwardメソッドにバイアス (+ b) を加算するロジックを追加。
# - これまで修正してきた lif_layer.py ではなく、spiking_transformer_v2.py が
#   実際にインポートしているこのファイルが、スパイクが発生しない根本原因であったため、
#   この修正により NEURON_BIAS: 2.0 が機能するようになります。
#
# 【!!! エラー修正 (log.txt) !!!】
# - ImportError: cannot import name 'get_neuron_by_name'
# - このモジュールを利用する他のファイル (spiking_transformer_v2.py など) が
#   必要とするファクトリ関数 `get_neuron_by_name` を追加。(L532)

from typing import Optional, Tuple, Any, List, cast, Dict, Type
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import surrogate, base # type: ignore[import-untyped]
import logging 

from .bif_neuron import BistableIFNeuron

logger = logging.getLogger(__name__)


class AdaptiveLIFNeuron(base.MemoryModule):
    """
    Adaptive Leaky Integrate-and-Fire (LIF) neuron with threshold adaptation.
    
    v2: Implements PLIF (Parametric LIF) by making tau_mem a learnable parameter.
    v6: Implements EL (Evolutionary Leak) [P2.1] by making tau_mem input-dependent.
    v_fix_bias_logic: Adds bias parameter (bias_init) and applies it in the forward pass.
    """
    log_tau_mem: nn.Parameter
    gate_tau_lin: Optional[nn.Linear]
    gate_input_proj_el: Optional[nn.Linear]
    bias: nn.Parameter # 【v_fix_bias_logic】 バイアスパラメータを追加

    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        base_threshold: float = 1.0,
        adaptation_strength: float = 0.1,
        target_spike_rate: float = 0.02,
        noise_intensity: float = 0.0,
        # 論文に基づく動的閾値パラメータを追加
        threshold_decay: float = 0.99,
        threshold_step: float = 0.05,
        # --- ▼ 改善 (v6): P2.1 (EL) 追加 ▼ ---
        evolutionary_leak: bool = False, # ELフラグ
        gate_input_features: Optional[int] = None, # GLIF互換
        # --- ▲ 改善 (v6) ▲ ---
        
        # --- ▼ 【v_fix_bias_logic】 バイアス引数を追加 ▼ ---
        bias_init: float = 0.0,
        # --- ▲ 【v_fix_bias_logic】 ▲ ---
        **kwargs: Any, # (v_init 互換性) v_init や他の引数を吸収
    ):
        super().__init__()
        self.features = features
        
        # --- (PLIF 修正済み) ---
        initial_log_tau = torch.full((features,), math.log(max(1.1, tau_mem - 1.1)))
        self.log_tau_mem = nn.Parameter(initial_log_tau)
        # --- (PLIF 修正済み) ---
        
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.noise_intensity = noise_intensity
        self.threshold_decay = threshold_decay
        self.threshold_step = threshold_step
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        # --- ▼ 【v_fix_bias_logic】 バイアスをパラメータとして初期化 ▼ ---
        self.bias = nn.Parameter(torch.full((features,), bias_init))
        # --- ▲ 【v_fix_bias_logic】 ▲ ---

        # (v_init 互換性) kwargs から v_init を取得
        self.v_init = float(kwargs.get('v_init', 0.0))

        # --- (v6: EL 修正済み) ---
        self.evolutionary_leak = evolutionary_leak
        self.gate_tau_lin = None
        self.gate_input_proj_el = None 
        if self.evolutionary_leak:
            if gate_input_features is None:
                gate_input_features = features
            self.gate_tau_lin = nn.Linear(gate_input_features, features)
            logger.info(f"AdaptiveLIFNeuron (features={features}): Evolutionary Leak (EL) [P2.1] ENABLED.")
        # --- (v6: EL 修正済み) ---

        self.register_buffer("mem", None)
        self.register_buffer("adaptive_threshold", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.mem = None
        self.adaptive_threshold = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Processes one timestep of input current."""
        if not self.stateful:
            self.mem = None
            self.adaptive_threshold = None

        # (v_init 互換性) v_init を使って初期化
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.full_like(x, self.v_init)
        if self.adaptive_threshold is None or self.adaptive_threshold.shape != x.shape:
            self.adaptive_threshold = torch.zeros_like(x)

        # --- ▼ 【v_fix_bias_logic】 バイアスを入力電流に加算 ▼ ---
        current_input = x + self.bias
        # --- ▲ 【v_fix_bias_logic】 ▲ ---

        # --- ▼ 改善 (v6): P2.1 (EL) 修正 ▼ ---
        mem_decay: torch.Tensor
        
        if self.evolutionary_leak and self.gate_tau_lin is not None:
            # EL (Evolutionary Leak)
            gate_input: torch.Tensor
            if current_input.shape[1] != self.gate_tau_lin.in_features: # x を current_input に変更
                 if self.gate_tau_lin.in_features == self.features:
                     gate_input = current_input # x を current_input に変更
                 else:
                     if not hasattr(self, 'gate_input_proj_el') or self.gate_input_proj_el is None:
                         self.gate_input_proj_el = nn.Linear(current_input.shape[1], self.gate_tau_lin.in_features).to(current_input.device) # x を current_input に変更
                     gate_input = self.gate_input_proj_el(current_input) # type: ignore[operator] # x を current_input に変更
            else:
                 gate_input = current_input # x を current_input に変更
            
            mem_decay_gate = torch.sigmoid(self.gate_tau_lin(gate_input))
            # 【v_fix_bias_logic】 x を current_input に変更
            self.mem = self.mem * mem_decay_gate + (1.0 - mem_decay_gate) * current_input
            
        else:
            # PLIF (Parametric LIF)
            current_tau_mem = torch.exp(self.log_tau_mem) + 1.1
            mem_decay = torch.exp(-1.0 / current_tau_mem)
            # 【v_fix_bias_logic】 x を current_input に変更
            self.mem = self.mem * mem_decay + current_input
        # --- ▲ 改善 (v6) ▲ ---
        
        if self.training and self.noise_intensity > 0:
            self.mem += torch.randn_like(self.mem) * self.noise_intensity
        
        self.adaptive_threshold = self.adaptive_threshold * self.threshold_decay
        current_threshold = self.base_threshold + self.adaptive_threshold
        spike = self.surrogate_function(self.mem - current_threshold)
        
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        reset_mask = spike.detach() 
        self.mem = self.mem * (1.0 - reset_mask) # ゼロリセット (修正不要)
        
        if self.training:
            self.adaptive_threshold = (
                self.adaptive_threshold + self.threshold_step * spike.detach()
            )
        else:
            with torch.no_grad():
                 self.adaptive_threshold = (
                    self.adaptive_threshold + self.threshold_step * spike
                )
        
        return spike, self.mem

    def get_spike_rate_loss(self) -> torch.Tensor:
        current_rate = self.spikes.mean()
        target = torch.tensor(self.target_spike_rate, device=current_rate.device)
        return F.mse_loss(current_rate, target)

class IzhikevichNeuron(base.MemoryModule):
    """
    Izhikevich neuron model, capable of producing a wide variety of firing patterns.
    """
    def __init__(
        self,
        features: int,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        dt: float = 0.5,
        **kwargs: Any, # (v_init 互換性)
    ):
        super().__init__()
        self.features = features
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        self.v_peak = 30.0
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.stateful = False
        
        # (v_init 互換性) v_init を c (v_rest) の代わりに使用
        self.v_init = float(kwargs.get('v_init', self.c))
        self.u_init = self.b * self.v_init

        self.register_buffer("v", None)
        self.register_buffer("u", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))

    def set_stateful(self, stateful: bool):
        """時系列データの処理モードを設定"""
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.v = None
        self.u = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Processes one timestep of input current with corrected Izhikevich dynamics.
        """
        if not self.stateful:
            self.v = None
            self.u = None
            
        # (v_init 互換性) v_init と u_init を使用
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.full_like(x, float(self.v_init))
        if self.u is None or self.u.shape != x.shape:
            self.u = torch.full_like(x, float(self.u_init))

        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + x
        du = self.a * (self.b * self.v - self.u)
        
        self.v = self.v + dv * self.dt
        self.u = self.u + du * self.dt
        
        spike = self.surrogate_function(self.v - self.v_peak)
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
        
        reset_mask = (self.v >= self.v_peak).detach()
        self.v = torch.where(reset_mask, torch.full_like(self.v, float(self.c)), self.v) # リセットは c (v_rest)
        self.u = torch.where(reset_mask, self.u + self.d, self.u)
        
        self.v = torch.clamp(self.v, min=-100.0, max=50.0)

        return spike, self.v
        
class ProbabilisticLIFNeuron(base.MemoryModule):
    """
    確率的にスパイクを生成する Leaky Integrate-and-Fire (LIF) ニューロン。
    論文 arXiv:2509.26507v1 のアイデアに基づく。
    v2: Implements PLIF by making tau_mem a learnable parameter.
    """
    log_tau_mem: nn.Parameter
    
    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        temperature: float = 0.5, # スパイク確率の鋭敏さを制御
        noise_intensity: float = 0.0,
        **kwargs: Any, # (v_init 互換性)
    ):
        super().__init__()
        self.features = features
        
        initial_log_tau = torch.full((features,), math.log(max(1.1, tau_mem - 1.1)))
        self.log_tau_mem = nn.Parameter(initial_log_tau)
        
        self.threshold = threshold
        self.temperature = temperature # 確率計算の温度パラメータ
        self.noise_intensity = noise_intensity
        
        # (v_init 互換性)
        self.v_init = float(kwargs.get('v_init', 0.0))

        self.register_buffer("mem", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        """時系列データの処理モードを設定"""
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        """Resets the neuron's state variables."""
        super().reset()
        self.mem = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Processes one timestep of input current and generates probabilistic spikes."""
        if not self.stateful:
            self.mem = None

        # (v_init 互換性)
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.full_like(x, self.v_init)

        current_tau_mem = torch.exp(self.log_tau_mem) + 1.1
        mem_decay = torch.exp(-1.0 / current_tau_mem)
        
        self.mem = self.mem * mem_decay + x

        if self.training and self.noise_intensity > 0:
            self.mem += torch.randn_like(self.mem) * self.noise_intensity

        spike_prob = torch.sigmoid((self.mem - self.threshold) / self.temperature)
        spike = (torch.rand_like(self.mem) < spike_prob).float()

        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike

        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        reset_mask = spike.detach()
        self.mem = self.mem * (1.0 - reset_mask)

        return spike, self.mem

    def get_spike_rate_loss(self) -> torch.Tensor:
        """確率的モデルではターゲットスパイク率の損失は通常適用しないが、互換性のために残す"""
        return torch.tensor(0.0, device=self.spikes.device)

class GLIFNeuron(base.MemoryModule):
    """
    Gated Leaky Integrate-and-Fire (GLIF) ニューロン。 (P2.2)
    設計思想.md (セクション3.1, 引用[8]) に基づき、
    膜時定数(tau)を、入力依存の「ゲート」によって
    動的に制御する学習可能なニューロンモデル。
    """
    base_threshold: nn.Parameter
    gate_tau_lin: nn.Linear
    v_reset: nn.Parameter # 学習可能なリセット電位
    # --- ▼ 改善 (v6): P2.1 (EL) との互換性 ▼ ---
    gate_input_proj_glif: Optional[nn.Linear]
    # --- ▲ 改善 (v6) ▲ ---

    def __init__(
        self,
        features: int,
        base_threshold: float = 1.0,
        gate_input_features: Optional[int] = None, # ゲート制御入力の次元 (Noneの場合はfeaturesと同じ)
        **kwargs: Any # 他のLIFパラメータ (tau_memなど) を無視
    ):
        super().__init__()
        self.features = features
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        
        gate_input_dim: int
        if gate_input_features is None:
            gate_input_dim = features
        else:
            gate_input_dim = gate_input_features
        
        # 1. 学習可能なリセット電位 (v_reset)
        # (引用[8]のGLIFはリセット機構も学習可能とする)
        self.v_reset = nn.Parameter(torch.full((features,), 0.0)) # 0.0 で初期化
        
        # (v_init 互換性)
        self.v_init = float(kwargs.get('v_init', 0.0))
        
        # 2. 膜時定数(tau)を制御するゲート (P2.2)
        # ゲートの入力次元を gate_input_features に設定
        self.gate_tau_lin = nn.Linear(gate_input_dim, features)
        self.gate_input_proj_glif = None # mypyのために初期化
        
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("mem", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.mem = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """1タイムステップの処理"""
        if not self.stateful:
            self.mem = None

        # (v_init 互換性)
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.full_like(x, self.v_init)
        
        # --- 1. ゲートの計算 (P2.2) ---
        # ゲート入力 (x) の次元チェック
        gate_input: torch.Tensor
        if x.shape[1] != self.gate_tau_lin.in_features:
             # GLIFのゲート入力次元と、ニューロンの入力次元が異なる場合
             if self.gate_tau_lin.in_features == self.features:
                 gate_input = x
             else:
                 #
                 # ゲート入力次元が異なる場合は、入力を射影するか、エラーを出す
                 # ここでは、入力 (x) をゲート入力次元に射影する (簡易的な実装)
                 # --- ▼ 改善 (v6): P2.1 (EL) との互換性 ▼ ---
                 if not hasattr(self, 'gate_input_proj_glif') or self.gate_input_proj_glif is None:
                     self.gate_input_proj_glif = nn.Linear(x.shape[1], self.gate_tau_lin.in_features).to(x.device)
                 gate_input = self.gate_input_proj_glif(x)
                 # --- ▲ 改善 (v6) ▲ ---
        else:
             gate_input = x
        
        # 時定数ゲート (mem_decay) を計算
        # Sigmoidの出力は (0, 1)。1に近いほど記憶が保持される (tau大)
        mem_decay_gate = torch.sigmoid(self.gate_tau_lin(gate_input))
        
        v_reset_gated = self.v_reset 

        # --- 2. 膜電位の更新 (LIFダイナミクス) ---
        # ゲートで制御された減衰
        self.mem = self.mem * mem_decay_gate + (1.0 - mem_decay_gate) * x
        
        # --- 3. スパイク生成 ---
        spike = self.surrogate_function(self.mem - self.base_threshold)
        
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        # --- 4. リセット ---
        reset_mask = spike.detach() 
        self.mem = self.mem * (1.0 - reset_mask) + reset_mask * v_reset_gated
        
        return spike, self.mem

# --- ▼▼▼ SNN5改善レポートに基づく追加実装 ▼▼▼ ---

class TC_LIF(base.MemoryModule):
    """
    Two-Compartment LIF (TC-LIF) ニューロン。
    SNN5改善レポート (セクション5.1, 引用[92, 94]) に基づく実装。
    長期時系列依存性の学習を目的とする。
    
    - Somatic (体細胞) コンパートメント: 高速なダイナミクス、スパイク生成
    - Dendritic (樹状突起) コンパートメント: 低速なダイナミクス、長期的な文脈の統合
    """
    log_tau_s: nn.Parameter # Somatic (体細胞) 膜時定数 (学習可能)
    log_tau_d: nn.Parameter # Dendritic (樹状突起) 膜時定数 (学習可能)
    w_ds: nn.Parameter # Dendritic -> Somatic 結合強度
    w_sd: nn.Parameter # Somatic -> Dendritic 結合強度 (フィードバック)

    def __init__(
        self,
        features: int,
        tau_s_init: float = 5.0,     # 体細胞の時定数 (高速)
        tau_d_init: float = 20.0,    # 樹状突起の時定数 (低速)
        w_ds_init: float = 0.5,      # 樹状突起から体細胞への結合強度
        w_sd_init: float = 0.1,      # 体細胞から樹状突起への結合強度
        base_threshold: float = 1.0,
        v_reset: float = 0.0,
        **kwargs: Any, # (v_init 互換性)
    ):
        super().__init__()
        self.features = features
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.v_reset = nn.Parameter(torch.full((features,), v_reset))
        
        # (v_init 互換性)
        self.v_init = float(kwargs.get('v_init', 0.0))
        
        # 学習可能な時定数 (PLIFと同様)
        self.log_tau_s = nn.Parameter(torch.full((features,), math.log(max(1.1, tau_s_init - 1.1))))
        self.log_tau_d = nn.Parameter(torch.full((features,), math.log(max(1.1, tau_d_init - 1.1))))
        
        # 学習可能な結合強度
        self.w_ds = nn.Parameter(torch.full((features,), w_ds_init))
        self.w_sd = nn.Parameter(torch.full((features,), w_sd_init))

        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("v_s", None) # Somatic membrane potential
        self.register_buffer("v_d", None) # Dendritic membrane potential
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.v_s = None
        self.v_d = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """1タイムステップの処理"""
        if not self.stateful:
            self.v_s = None
            self.v_d = None

        # (v_init 互換性)
        if self.v_s is None or self.v_s.shape != x.shape:
            self.v_s = torch.full_like(x, self.v_init)
        if self.v_d is None or self.v_d.shape != x.shape:
            self.v_d = torch.full_like(x, self.v_init)

        # 時定数から減衰率を計算
        current_tau_s = torch.exp(self.log_tau_s) + 1.1
        decay_s = torch.exp(-1.0 / current_tau_s)
        
        current_tau_d = torch.exp(self.log_tau_d) + 1.1
        decay_d = torch.exp(-1.0 / current_tau_d)

        # 1. 樹状突起 (Dendritic) の更新 (低速)
        # 入力電流(x)と体細胞からのフィードバック(w_sd * spikes)を受け取る
        # (スパイクではなく、結合された入力としてxを受け取る)
        # 論文[94]の実装に合わせ、xは両方のコンパートメントに直接入力されると仮定
        dendritic_input = x + self.w_sd * self.spikes # type: ignore[has-type]
        self.v_d = self.v_d * decay_d + dendritic_input
        
        # 2. 体細胞 (Somatic) の更新 (高速)
        somatic_input = x + self.w_ds * self.v_d # 樹状突起の電位で変調
        self.v_s = self.v_s * decay_s + somatic_input
        
        # 3. スパイク生成
        spike = self.surrogate_function(self.v_s - self.base_threshold)
        
        self.spikes = spike.detach() # 次のステップのフィードバック用 (勾配なし)
        with torch.no_grad():
            self.total_spikes += self.spikes.sum()

        # 4. リセット (v_resetは学習可能)
        reset_mask = self.spikes
        self.v_s = self.v_s * (1.0 - reset_mask) + reset_mask * self.v_reset
        # (論文[94]によっては樹状突起もリセットするが、ここでは体細胞のみ)
        
        return spike, self.v_s # 体細胞の電位を返す

class DualThresholdNeuron(base.MemoryModule):
    """
    Dual Threshold Neuron (エラー補償学習用)。
    SNN5改善レポート (セクション3.1, 引用[6]) に基づく実装。
    量子化エラーと不均一性エラーを削減する。
    v2: Implements PLIF by making tau_mem a learnable parameter.
    """
    log_tau_mem: nn.Parameter
    threshold_high: nn.Parameter # T_h (学習可能なしきい値)
    threshold_low: nn.Parameter  # T_l (デュアルしきい値)
    
    spikes: Tensor
    
    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        threshold_high_init: float = 1.0, # T_h (クリッピング用)
        threshold_low_init: float = 0.5,  # T_l (量子化エラー削減用)
        v_reset: float = 0.0,
        **kwargs: Any, # (v_init 互換性)
    ):
        super().__init__()
        self.features = features
        self.log_tau_mem = nn.Parameter(torch.full((features,), math.log(max(1.1, tau_mem - 1.1))))
        
        # 引用[6]に基づき、2つのしきい値を学習可能パラメータとする
        self.threshold_high = nn.Parameter(torch.full((features,), threshold_high_init))
        self.threshold_low = nn.Parameter(torch.full((features,), threshold_low_init))
        
        self.v_reset = nn.Parameter(torch.full((features,), v_reset))
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        
        # (v_init 互換性)
        self.v_init = float(kwargs.get('v_init', 0.0))

        self.register_buffer("mem", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.mem = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """1タイムステップの処理"""
        if not self.stateful:
            self.mem = None

        if self.mem is None or self.mem.shape != x.shape:
            # (v_init 互換性)
            # 引用[6]のT_l/2の代わりにv_initを使用（v_initが設定されていれば）
            init_val = self.v_init if self.v_init != 0.0 else (self.threshold_low.detach() / 2.0)
            self.mem = init_val.expand_as(x) # type: ignore[attr-defined]

        current_tau_mem = torch.exp(self.log_tau_mem) + 1.1
        mem_decay = torch.exp(-1.0 / current_tau_mem)
        
        # 膜電位の更新
        self.mem = self.mem * mem_decay + x
        
        # スパイク生成 (T_h を使用)
        spike_untyped = self.surrogate_function(self.mem - self.threshold_high)
        spike: Tensor = cast(Tensor, spike_untyped)
        
        current_spikes_detached: Tensor = spike.detach()
        
        if current_spikes_detached.ndim > 1:
            self.spikes = current_spikes_detached.mean(dim=0)
        else:
            self.spikes = current_spikes_detached

        with torch.no_grad():
            self.total_spikes += current_spikes_detached.sum() # type: ignore[has-type]
        
        # リセット (デュアルしきい値を使用)
        # 引用[6]の式(7)に基づくリセット
        # S=1 の場合: V[t+1] = V[t] - T_h
        # S=0 の場合: V[t+1] = V[t]
        # ただし、 V[t+1] < T_l の場合は、V[t+1] = V_reset (または T_l/2) にリセット
        
        reset_mem = self.mem - current_spikes_detached * self.threshold_high
        
        # T_l を下回ったニューロンを検出
        below_low_threshold = reset_mem < self.threshold_low
        
        reset_condition = (current_spikes_detached > 0.5) | below_low_threshold
        
        self.mem = torch.where(
            reset_condition,
            self.v_reset.expand_as(self.mem), # V_reset にリセット
            reset_mem # それ以外は減算後の膜電位を維持
        )
        
        return spike, self.mem

class ScaleAndFireNeuron(base.MemoryModule):
    """
    Scale-and-Fire (SFN) ニューロン。
    SNN5改善レポート (セクション3.2, 引用[18]) に基づく実装。
    T=1 (シングルタイムステップ) でのANN-SNN変換を目指す。
    
    これは時間ループを必要とせず、T=1で空間的なマルチしきい値計算を行う。
    """
    def __init__(
        self,
        features: int,
        num_levels: int = 8, # 空間的な量子化レベル数 (しきい値の数)
        base_threshold: float = 1.0,
        **kwargs: Any, # (v_init 互換性)
    ):
        super().__init__()
        self.features = features
        self.num_levels = num_levels # K (しきい値の数)
        
        # K個の学習可能なしきい値を定義
        thresholds = torch.linspace(0.5, num_levels - 0.5, num_levels) / num_levels * base_threshold
        self.thresholds = nn.Parameter(thresholds.unsqueeze(0).repeat(features, 1)) # (Features, K)
        
        # K個の学習可能なスケーリング係数（重み）を定義
        self.scales = nn.Parameter(torch.ones(features, num_levels)) # (Features, K)
        
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))

    def set_stateful(self, stateful: bool):
        # SFNはステートレス (T=1)
        pass

    def reset(self):
        super().reset()
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        T=1 でのアナログ入力処理 (ANN-SNN変換用)。
        
        Args:
            x (Tensor): ANNからのアナログ活性化入力 (B, Features)。
            
        Returns:
            Tuple[Tensor, Tensor]: (SNNの出力値(アナログ), 膜電位(ダミー))
        """
        B, N = x.shape
        if N != self.features:
            raise ValueError(f"Input dimension ({N}) does not match num_neurons ({self.features})\")")

        # 入力を (B, N, 1) に拡張し、しきい値 (N, K) と比較
        x_expanded = x.unsqueeze(-1) # (B, N, 1)
        thresholds_expanded = self.thresholds.unsqueeze(0) # (1, N, K)
        
        # 空間的なマルチしきい値比較 (B, N, K)
        spatial_spikes = self.surrogate_function(x_expanded - thresholds_expanded)
        
        # 学習可能なスケーリング係数（重み）を適用
        scales_expanded = self.scales.unsqueeze(0) # (1, N, K)
        
        # 空間的なスパイクに重み付けして合計 (B, N)
        # これが T=1 での時間積分（発火率）の空間的近似値となる
        output_analog = (spatial_spikes * scales_expanded).sum(dim=-1)
        
        # スパイク統計（ダミー）
        self.spikes = spatial_spikes.mean(dim=(0, 2)) # Kレベルでの平均スパイク率
        with torch.no_grad():
            self.total_spikes += spatial_spikes.detach().sum()

        # SFNはT=1でアナログ値を直接出力する (SNNの最終層や変換層として使う)
        # 戻り値の型 (spike, mem) に合わせる
        return output_analog, output_analog # memの代わりにoutput_analogを返す


__all__ = [
    "AdaptiveLIFNeuron",
    "IzhikevichNeuron",
    "ProbabilisticLIFNeuron",
    "GLIFNeuron",
    "TC_LIF",
    "DualThresholdNeuron",
    "ScaleAndFireNeuron",
    "BistableIFNeuron"
]

# --- ▼▼▼ 【!!! エラー修正 (log.txt) !!!】 ▼▼▼
# (log.txt の ImportError を解決するため、ファクトリ関数を追加)

# ニューロンのタイプ名 (文字列) とクラスをマッピング
NEURON_REGISTRY: Dict[str, Type[base.MemoryModule]] = {
    # (注: このファイル内のクラス定義に合わせて 'features' 引数を必須とする)
    "lif": AdaptiveLIFNeuron,
    "bif": BistableIFNeuron,
    "izhikevich": IzhikevichNeuron,
    "glif": GLIFNeuron,
    "tc_lif": TC_LIF,
    "dual_threshold": DualThresholdNeuron,
    "scale_and_fire": ScaleAndFireNeuron,
    "probabilistic_lif": ProbabilisticLIFNeuron,
}

def get_neuron_by_name(name: str, params: Dict[str, Any]) -> base.MemoryModule:
    """
    ニューロンのタイプ名 (文字列) に基づいて、
    ニューロンクラスのインスタンスを作成して返します。
    
    Args:
        name (str): ニューロンのタイプ名 (例: "lif")。
        params (Dict[str, Any]): ニューロンのコンストラクタに渡すパラメータ辞書。
        
    Returns:
        base.MemoryModule: インスタンス化されたニューロン。
        
    Raises:
        ValueError: 指定された 'name' がレジストリにない場合。
    """
    name_lower = name.lower()
    if name_lower not in NEURON_REGISTRY:
        raise ValueError(
            f"Unknown neuron type: '{name}'. "
            f"Available types: {list(NEURON_REGISTRY.keys())}"
        )
        
    NeuronClass = NEURON_REGISTRY[name_lower]
    
    # (v_fix_type_error / v_fix_import_error 互換性)
    # spiking_transformer_v2.py は 'd_model' または 'dim_feedforward' を
    # ニューロンの 'features' 引数として期待している可能性がある。
    # 'features' が params にない場合、d_model や d_ff から推測する。
    if 'features' not in params:
        if 'd_model' in params:
            params['features'] = int(params['d_model'])
        elif 'dim_feedforward' in params:
            params['features'] = int(params['dim_feedforward'])
        else:
            # features が見つからず、NeuronClass が features を必要とするかチェック
            import inspect
            sig = inspect.signature(NeuronClass.__init__)
            if 'features' in sig.parameters:
                 logger.warning(
                     f"Neuron type '{name_lower}' requires 'features', but it was not found in params. "
                     f"This might cause an error. Params provided: {list(params.keys())}"
                 )

    
    try:
        # パラメータ辞書を渡してインスタンス化
        return NeuronClass(**params)
    except TypeError as e:
        logger.error(
            f"Failed to instantiate neuron '{name_lower}' with params: {params}. "
            f"Error: {e}"
        )
        # (デバッグ用) 期待される引数と渡された引数のミスマッチの詳細を出力
        import inspect
        sig = inspect.signature(NeuronClass.__init__)
        expected_params = list(sig.parameters.keys())
        provided_params = list(params.keys())
        logger.error(f"'{name_lower}' __init__ expects: {expected_params}")
        logger.error(f"Params provided: {provided_params}")
        
        # 'features' がないのが原因の場合のエラーメッセージ
        if 'features' in expected_params and 'features' not in provided_params:
            logger.error(
                "Critical Error: 'features' (e.g., d_model) was not passed to the neuron constructor."
            )
        raise e

# --- ▲▲▲ 【!!! エラー修正 (log.txt) !!!】 ▲▲▲
