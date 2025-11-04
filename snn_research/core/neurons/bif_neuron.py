# ファイルパス: snn_research/core/neurons/bif_neuron.py
# Title: Bistable Integrate-and-Fire (BIF) ニューロンモデル (完成版)
# Description: Improvement-Plan.md に基づき、双安定性を持つBIFニューロンを実装。
#              _bif_dynamicsメソッドに具体的な三次非線形項を含む更新式を実装。
#
# 改善 (v2):
# - spikingjelly.activation_based.base.MemoryModule を継承。
# - set_stateful および reset メソッドを実装し、functional.reset_net に対応。

import torch
import torch.nn as nn
from typing import Tuple, Optional, cast, Any

# 代理勾配関数 (学習に必要)
from spikingjelly.activation_based import surrogate, base # type: ignore[import-untyped]
import logging # ロギングを追加

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ▼ 改善 (v2): MemoryModule を継承 ▼ ---
class BistableIFNeuron(base.MemoryModule):
# --- ▲ 改善 (v2) ▲ ---
    """
    双安定積分発火ニューロン (Bistable Integrate-and-Fire Neuron) の改善版実装。
    (課題はdocstringに記載されているものとする)
    """
    # バッファの型ヒントを追加
    membrane_potential: Optional[torch.Tensor]
    spikes: torch.Tensor # type: ignore[assignment]

    def __init__(self,
                 features: int,
                 v_threshold_high: float = 1.0,
                 v_reset: float = 0.6,
                 tau_mem: float = 10.0,
                 bistable_strength: float = 0.25,
                 v_rest: float = 0.0,
                 unstable_equilibrium_offset: float = 0.5,
                 surrogate_function: nn.Module = surrogate.ATan(alpha=2.0) # type: ignore[assignment]
                ):
        super().__init__()
        self.features = features
        self.v_th_high = v_threshold_high
        self.v_reset = v_reset
        self.tau_mem = tau_mem
        self.dt = 1.0
        self.decay = torch.exp(torch.tensor(-self.dt / self.tau_mem))
        self.bistable_strength = bistable_strength
        self.v_rest = v_rest
        self.unstable_equilibrium = self.v_rest + unstable_equilibrium_offset
        self.surrogate_function = surrogate_function
        self.stateful = False

        # 状態変数 (初期化戦略)
        initial_potential = torch.randn(features) * 0.05 + self.unstable_equilibrium
        self.register_buffer("membrane_potential", initial_potential)
        self.register_buffer("spikes", torch.zeros(features)) # 現在のタイムステップのスパイク
        # --- ▼ 改善 (v2): total_spikes を追加 ▼ ---
        self.register_buffer("total_spikes", torch.tensor(0.0))
        # --- ▲ 改善 (v2) ▲ ---

        logging.info("BistableIFNeuron: ダイナミクス実装を改善しました。")
        logging.warning(f"  - Parameters requiring careful tuning: v_reset={v_reset}, bistable_strength={bistable_strength}, unstable_equilibrium={self.unstable_equilibrium}")

    # --- ▼ 改善 (v2): set_stateful と reset を実装 ▼ ---
    def set_stateful(self, stateful: bool) -> None:
        """時系列データの処理モードを設定"""
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self) -> None:
        """状態変数をリセットします。"""
        super().reset() # MemoryModuleのresetを呼ぶ
        if self.membrane_potential is not None:
            noise = torch.randn_like(self.membrane_potential) * 0.05
            self.membrane_potential = noise + self.unstable_equilibrium
        if hasattr(self, 'spikes') and self.spikes is not None:
             self.spikes.zero_()
        if hasattr(self, 'total_spikes') and self.total_spikes is not None:
             self.total_spikes.zero_()
    # --- ▲ 改善 (v2) ▲ ---

    def _bif_dynamics(self, v: torch.Tensor, input_current: torch.Tensor) -> torch.Tensor:
        """
        BIFニューロンの膜電位更新式（三次非線形項を含む）。
        """
        leak = (v - self.v_rest) / self.tau_mem
        non_linear = self.bistable_strength * (v - self.v_rest) * (v - self.unstable_equilibrium) * (self.v_th_high - v)
        dv = (-leak + non_linear + input_current) * self.dt
        return v + dv

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # type: ignore[override]
        """
        1タイムステップ分の処理を実行します。
        """
        current_membrane_potential: torch.Tensor
        
        # 状態の初期化または取得
        if not self.stateful:
            self.reset()
        
        if self.membrane_potential is None or self.membrane_potential.shape != input_current.shape or self.membrane_potential.device != input_current.device:
             noise = torch.randn_like(input_current) * 0.05
             current_membrane_potential = (noise + self.unstable_equilibrium).to(input_current.device)
        else:
             current_membrane_potential = self.membrane_potential

        # --- BIFの膜電位更新 ---
        next_potential = self._bif_dynamics(current_membrane_potential, input_current)

        # スパイク判定 (代理勾配を使用)
        spike = self.surrogate_function(next_potential - self.v_th_high)

        # リセット処理 (スパイクしたニューロンのみ)
        reset_potential = torch.where(spike > 0.5, torch.full_like(next_potential, self.v_reset), next_potential)

        # 内部状態の更新 (statefulな場合)
        if self.stateful:
            self.membrane_potential = reset_potential
        
        self.spikes = spike # 学習用に勾配を持つスパイクを保持
        
        # --- ▼ 改善 (v2): total_spikes を記録 ▼ ---
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
        # --- ▲ 改善 (v2) ▲ ---

        return spike, reset_potential
