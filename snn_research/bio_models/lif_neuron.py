# ファイルパス: snn_research/bio_models/lif_neuron.py
# (改修: P8.3 適応的閾値の実装)
#
# Title: Leaky Integrate-and-Fire (LIF) ニューロンモデル
# Description: シンプルなLIFニューロンを実装します。
#
# 改善 (v2):
# - doc/ROADMAP.md (P8.3) に基づき、恒常性維持（ホメオスタシス）のための
#   適応的発火閾値メカニズムを実装。
# - AdaptiveLIFNeuron のロジックを参考に、
#   `threshold_decay` と `threshold_step` を導入。

import torch
import torch.nn as nn
# --- ▼ 改善 (v2): 型ヒントを追加 ▼ ---
from typing import Dict, Any, Optional
# --- ▲ 改善 (v2) ▲ ---

class BioLIFNeuron(nn.Module):
    """
    生物学的学習則のための適応的閾値を持つLIFニューロン (P8.3)。
    """
    # --- ▼ 改善 (v2): 状態変数の型ヒントを追加 ▼ ---
    voltages: torch.Tensor
    adaptive_threshold: torch.Tensor
    # --- ▲ 改善 (v2) ▲ ---

    def __init__(
        self, 
        n_neurons: int, 
        neuron_params: Dict[str, Any], 
        dt: float = 1.0
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau_mem = neuron_params.get('tau_mem', 10.0)
        self.v_thresh_base = neuron_params.get('v_threshold', 1.0)
        self.v_reset = neuron_params.get('v_reset', 0.0)
        self.v_rest = neuron_params.get('v_rest', 0.0)
        self.dt = dt
        
        # --- ▼ 改善 (v2): P8.3 適応的閾値パラメータ ▼ ---
        self.threshold_decay = neuron_params.get('threshold_decay', 0.99) # 減衰係数
        self.threshold_step = neuron_params.get('threshold_step', 0.05)   # 発火時の上昇量
        # --- ▲ 改善 (v2) ▲ ---
        
        self.voltages = torch.full((n_neurons,), self.v_rest)
        # --- ▼ 改善 (v2): P8.3 適応的閾値バッファ ▼ ---
        self.adaptive_threshold = torch.zeros(n_neurons)
        # --- ▲ 改善 (v2) ▲ ---

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        if self.voltages.device != input_current.device:
            self.voltages = self.voltages.to(input_current.device)
        # --- ▼ 改善 (v2): P8.3 適応的閾値バッファのデバイス同期 ▼ ---
        if self.adaptive_threshold.device != input_current.device:
            self.adaptive_threshold = self.adaptive_threshold.to(input_current.device)
        # --- ▲ 改善 (v2) ▲ ---

        # 膜電位の漏れ
        leak = (self.voltages - self.v_rest) / self.tau_mem
        
        # 膜電位の更新
        self.voltages += (-leak + input_current) * self.dt
        
        # --- ▼ 改善 (v2): P8.3 適応的閾値の適用 ▼ ---
        # 1. 適応的閾値を減衰させる
        self.adaptive_threshold *= self.threshold_decay
        
        # 2. 現在の閾値を計算
        current_threshold = self.v_thresh_base + self.adaptive_threshold
        
        # 3. 発火判定
        spikes = (self.voltages >= current_threshold).float()
        
        # 4. 発火したニューロンの閾値を上昇させる
        self.adaptive_threshold += spikes * self.threshold_step
        # --- ▲ 改善 (v2) ▲ ---
        
        # リセット
        self.voltages[spikes.bool()] = self.v_reset
        
        return spikes