# ファイルパス: snn_research/core/neurons/bif_neuron.py
# Title: Bistable Integrate-and-Fire (BIF) ニューロンモデル (ダイナミクス実装改善)
# Description: Improvement-Plan.md に基づき、双安定性を持つBIFニューロンを実装。
#              _bif_dynamicsメソッドに具体的な三次非線形項を含む更新式を実装し、
#              簡易実装から品質を向上させる。パラメータ調整の課題はコメントで明記。

import torch
import torch.nn as nn
from typing import Tuple, Optional

# 代理勾配関数 (学習に必要)
from spikingjelly.activation_based import surrogate # type: ignore
import logging # ロギングを追加

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BistableIFNeuron(nn.Module):
    """
    双安定積分発火ニューロン (Bistable Integrate-and-Fire Neuron) の改善版実装。

    Improvement-Plan.mdに基づく課題:
    - ⚠️ 課題1: 双安定領域のパラメータ調整が極めて困難。特に v_reset と bistable_strength (bパラメータ相当) の相互作用が複雑。
               v_reset > sqrt(|b|) のような条件を満たす必要がある場合がある。
    - ⚠️ 課題2: 初期条件依存性。膜電位の初期値によって、同じ入力でも発火/静止の挙動が変わる。不安定平衡点周辺の初期化が推奨されるが最適値は不明。
    - ⚠️ 課題3: バックプロパゲーションの不安定性。双安定領域付近では勾配が消失または爆発しやすい。代理勾配関数の選択と調整が重要。
    - ⚠️ 課題4: 静止状態へのトラップ。学習初期にニューロンが発火せず学習が進まない可能性。
    - ⚠️ 課題5: 発火状態からの暴走。一度発火すると止まらなくなる可能性。適切な抑制メカニズムが必要。
    """
    # バッファの型ヒントを追加
    membrane_potential: torch.Tensor
    spikes: torch.Tensor

    def __init__(self,
                 features: int,
                 v_threshold_high: float = 1.0,
                 # v_threshold_low: float = -0.5, # 静止状態への遷移閾値 (下限) - 実装を簡略化するため削除
                 v_reset: float = 0.6,          # スパイク後のリセット電位 ★課題1: 要調整
                 tau_mem: float = 10.0,         # 膜時定数 (ms)
                 bistable_strength: float = 0.25, # 双安定性の強さ (bパラメータ相当) ★課題1: 要調整
                 v_rest: float = 0.0,           # 静止電位
                 unstable_equilibrium_offset: float = 0.5, # 不安定平衡点の位置 (v_restからのオフセット) ★課題1,2: 要調整
                 surrogate_function = surrogate.ATan(alpha=2.0) # 代理勾配関数 ★課題3: 要調整
                ):
        """
        Args:
            features (int): ニューロン数。
            v_threshold_high (float): 発火閾値 (上側)。
            v_reset (float): リセット電位。
            tau_mem (float): 膜時定数。
            bistable_strength (float): 双安定性の強さ。
            v_rest (float): 静止電位。
            unstable_equilibrium_offset (float): 不安定平衡点のv_restからのオフセット。
            surrogate_function: 代理勾配関数。
        """
        super().__init__()
        self.features = features
        self.v_th_high = v_threshold_high
        # self.v_th_low = v_threshold_low
        self.v_reset = v_reset
        self.tau_mem = tau_mem
        self.dt = 1.0 # 仮の時間ステップ (ms)
        self.decay = torch.exp(torch.tensor(-self.dt / self.tau_mem))
        self.bistable_strength = bistable_strength
        self.v_rest = v_rest
        # 不安定平衡点の計算
        self.unstable_equilibrium = self.v_rest + unstable_equilibrium_offset
        self.surrogate_function = surrogate_function

        # 状態変数 (初期化戦略)
        # ★課題2: 不安定平衡点の周辺で初期化
        initial_potential = torch.randn(features) * 0.05 + self.unstable_equilibrium
        self.register_buffer("membrane_potential", initial_potential)
        self.register_buffer("spikes", torch.zeros(features)) # 現在のタイムステップのスパイク

        logging.info("BistableIFNeuron: ダイナミクス実装を改善しました。")
        logging.warning(f"  - Parameters requiring careful tuning: v_reset={v_reset}, bistable_strength={bistable_strength}, unstable_equilibrium={self.unstable_equilibrium}")

    def _bif_dynamics(self, v: torch.Tensor, input_current: torch.Tensor) -> torch.Tensor:
        """
        BIFニューロンの膜電位更新式（三次非線形項を含む）。
        dv/dt = -(v - v_rest)/tau + b * (v - v_rest) * (v - v_unstable) * (v_th_high - v) + I
        """
        # リーク項
        leak = (v - self.v_rest) / self.tau_mem
        # 双安定性を生む三次非線形項
        # 閾値より大幅に超えた場合に発散しないようにクリップするなどの工夫が必要な場合がある
        non_linear = self.bistable_strength * (v - self.v_rest) * (v - self.unstable_equilibrium) * (self.v_th_high - v)
        # 更新
        dv = (-leak + non_linear + input_current) * self.dt
        return v + dv

    def forward(self, input_current: torch.Tensor, membrane_potential_in: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1タイムステップ分の処理を実行します。
        """
        current_membrane_potential: torch.Tensor
        # 状態の初期化または取得
        if membrane_potential_in is None:
            if not hasattr(self, 'membrane_potential') or self.membrane_potential is None or self.membrane_potential.shape != input_current.shape or self.membrane_potential.device != input_current.device:
                 # 形状やデバイスが異なる場合、または初期化されていない場合は初期化
                 noise = torch.randn_like(input_current) * 0.05
                 self.membrane_potential = noise + self.unstable_equilibrium
            current_membrane_potential = self.membrane_potential
        else:
            current_membrane_potential = membrane_potential_in

        # --- BIFの膜電位更新 ---
        next_potential = self._bif_dynamics(current_membrane_potential, input_current)

        # スパイク判定 (代理勾配を使用)
        # ★課題3: 代理勾配の選択とパラメータ調整が重要
        spike = self.surrogate_function(next_potential - self.v_th_high)

        # リセット処理 (スパイクしたニューロンのみ)
        # スパイクしたら v_reset に、そうでなければ更新後の電位を保持
        reset_potential = torch.where(spike > 0.5, torch.full_like(next_potential, self.v_reset), next_potential)

        # 内部状態の更新 (statefulな場合)
        if membrane_potential_in is None:
            self.membrane_potential = reset_potential

        self.spikes = spike # 学習用に勾配を持つスパイクを保持

        # 戻り値としては勾配のないスパイク (detach) と更新後の膜電位を返すことが多い
        return spike.detach(), reset_potential

    def reset(self):
        """状態変数をリセットします。"""
        if hasattr(self, 'membrane_potential') and self.membrane_potential is not None:
             noise = torch.randn_like(self.membrane_potential) * 0.05
             # ★課題2: 不安定平衡点の周辺にリセット
             self.membrane_potential = noise + self.unstable_equilibrium
        if hasattr(self, 'spikes') and self.spikes is not None:
             self.spikes.zero_()