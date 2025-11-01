# ファイルパス: snn_research/core/neurons.py
# (修正)
# 修正: 論文「Dynamic Threshold and Multi-level Attention」に基づき、
#       AdaptiveLIFNeuronに動的発火閾値メカナイズムを導入。
# 改善(snn_4_ann_parity_plan):
# - IzhikevichNeuronにset_statefulメソッドを追加し、AdaptiveLIFNeuronとの
#   インターフェース互換性を確保。これにより、SpikingTransformerなどで
#   ニューロンモデルを切り替え可能にする。
#
# 改善 (v2):
# - doc/SNN開発：基本設計思想.md (セクション3.1, 引用[7]) に基づき、
#   AdaptiveLIFNeuronの膜時定数(tau_mem)を学習可能なパラメータに変更 (PLIF化)。
#
# 改善 (v3):
# - doc/SNN開発：基本設計思想.md (セクション3.1, 引用[8]) に基づき、
#   GLIF (Gated LIF) ニューロンを新規追加。
#
# 修正 (v4):
# - mypy [name-defined] エラーを解消するため、Any をインポート。

# --- ▼ 修正 ▼ ---
from typing import Optional, Tuple, Any
# --- ▲ 修正 ▲ ---
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import surrogate, base # type: ignore

class AdaptiveLIFNeuron(base.MemoryModule):
    """
    Adaptive Leaky Integrate-and-Fire (LIF) neuron with threshold adaptation.
    Designed for vectorized operations and to be BPTT-friendly.
    
    v2: Implements PLIF (Parametric LIF) by making tau_mem a learnable parameter.
    """
    tau_mem: nn.Parameter

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
    ):
        super().__init__()
        self.features = features
        
        initial_log_tau = torch.full((features,), math.log(max(1.1, tau_mem - 1.1)))
        self.log_tau_mem = nn.Parameter(initial_log_tau)
        
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.noise_intensity = noise_intensity
        self.threshold_decay = threshold_decay
        self.threshold_step = threshold_step
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("mem", None)
        self.register_buffer("adaptive_threshold", None)
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
        self.adaptive_threshold = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Processes one timestep of input current."""
        if not self.stateful:
            self.mem = None
            self.adaptive_threshold = None

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        if self.adaptive_threshold is None or self.adaptive_threshold.shape != x.shape:
            self.adaptive_threshold = torch.zeros_like(x)

        current_tau_mem = torch.exp(self.log_tau_mem) + 1.1
        mem_decay = torch.exp(-1.0 / current_tau_mem)
        
        self.mem = self.mem * mem_decay + x
        
        if self.training and self.noise_intensity > 0:
            self.mem += torch.randn_like(self.mem) * self.noise_intensity
        
        self.adaptive_threshold = self.adaptive_threshold * self.threshold_decay
        current_threshold = self.base_threshold + self.adaptive_threshold
        spike = self.surrogate_function(self.mem - current_threshold)
        
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        reset_mask = spike.detach() 
        self.mem = self.mem * (1.0 - reset_mask)
        
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
        """スパイク率の目標値からの乖離を損失として返す"""
        current_rate = self.spikes.mean()
        target = torch.tensor(self.target_spike_rate, device=current_rate.device)
        return F.mse_loss(current_rate, target)

class IzhikevichNeuron(base.MemoryModule):
    # (変更なし)
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
            
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.full_like(x, self.c)
        if self.u is None or self.u.shape != x.shape:
            self.u = torch.full_like(x, self.b * self.c)

        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + x
        du = self.a * (self.b * self.v - self.u)
        
        self.v = self.v + dv * self.dt
        self.u = self.u + du * self.dt
        
        spike = self.surrogate_function(self.v - self.v_peak)
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
        
        reset_mask = (self.v >= self.v_peak).detach()
        self.v = torch.where(reset_mask, torch.full_like(self.v, self.c), self.v)
        self.u = torch.where(reset_mask, self.u + self.d, self.u)
        
        self.v = torch.clamp(self.v, min=-100.0, max=50.0)

        return spike, self.v
        
class ProbabilisticLIFNeuron(base.MemoryModule):
    """
    確率的にスパイクを生成する Leaky Integrate-and-Fire (LIF) ニューロン。
    論文 arXiv:2509.26507v1 のアイデアに基づく。
    """
    log_tau_mem: nn.Parameter
    
    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        temperature: float = 0.5, # スパイク確率の鋭敏さを制御
        noise_intensity: float = 0.0,
    ):
        super().__init__()
        self.features = features
        
        initial_log_tau = torch.full((features,), math.log(max(1.1, tau_mem - 1.1)))
        self.log_tau_mem = nn.Parameter(initial_log_tau)
        
        self.threshold = threshold
        self.temperature = temperature # 確率計算の温度パラメータ
        self.noise_intensity = noise_intensity

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

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)

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
    Gated Leaky Integrate-and-Fire (GLIF) ニューロン。
    設計思想.md (セクション3.1, 引用[8]) に基づき、
    膜時定数(tau)を、入力依存の「ゲート」によって
    動的に制御する学習可能なニューロンモデル。
    """
    base_threshold: nn.Parameter
    gate_tau_lin: nn.Linear
    v_reset: nn.Parameter # 学習可能なリセット電位

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
        
        if gate_input_features is None:
            gate_input_features = features
        
        # 1. 学習可能なリセット電位 (v_reset)
        # (引用[8]のGLIFはリセット機構も学習可能とする)
        self.v_reset = nn.Parameter(torch.full((features,), 0.0)) # 0.0 で初期化
        
        # 2. 膜時定数(tau)を制御するゲート
        # ゲートの入力次元を gate_input_features に設定
        self.gate_tau_lin = nn.Linear(gate_input_features, features)
        
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

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        
        # --- 1. ゲートの計算 ---
        # ゲート入力 (x) の次元チェック
        if x.shape[1] != self.gate_tau_lin.in_features:
             # GLIFのゲート入力次元と、ニューロンの入力次元が異なる場合
             # (例: SimpleSNNで d_model -> hidden_size に射影された場合)
             # gate_input_features が features (hidden_size) と同じなら x を使う
             if self.gate_tau_lin.in_features == self.features:
                 gate_input = x
             else:
                 raise ValueError(f"GLIF gate input dim mismatch. Expected {self.gate_tau_lin.in_features} but got {x.shape[1]}")
        else:
             gate_input = x
        
        # 時定数ゲート (mem_decay) を計算
        mem_decay_gate = torch.sigmoid(self.gate_tau_lin(gate_input))
        
        v_reset_gated = self.v_reset 

        # --- 2. 膜電位の更新 (LIFダイナミクス) ---
        self.mem = self.mem * mem_decay_gate + x
        
        # --- 3. スパイク生成 ---
        spike = self.surrogate_function(self.mem - self.base_threshold)
        
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        # --- 4. リセット ---
        reset_mask = spike.detach() 
        self.mem = self.mem * (1.0 - reset_mask) + reset_mask * v_reset_gated
        
        return spike, self.mem
