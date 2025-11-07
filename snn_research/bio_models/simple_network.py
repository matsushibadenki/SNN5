# snn_research/bio_models/simple_network.py
# (修正)
# 修正: learning_rule.update がタプルを返すようになったため、
#       戻り値を正しくアンパックして使用する。
# 修正: CausalTraceCreditAssignmentEnhancedV2 に対応
#
# 改善 (v2):
# - doc/The-flow-of-brain-behavior.md および doc/プロジェクト強化案の調査.md (セクション2.3) に基づき、
#   単一の学習則しか持てなかった制約を解消。
# - シナプス可塑性ルール (synaptic_rule) と 恒常性維持ルール (homeostatic_rule) を
#   別々に受け取り、両方を適用できるように __init__ と update_weights を変更。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

from .lif_neuron import BioLIFNeuron
from snn_research.learning_rules.base_rule import BioLearningRule
# --- ▼ 修正 ▼ ---
# V2 クラスをインポート
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignmentEnhancedV2
# --- ▲ 修正 ▲ ---


class BioSNN(nn.Module):
    # ... (init, forward は変更なし) ...
    # --- ▼ 改善 (v2): __init__ のシグネチャを変更 ▼ ---
    def __init__(
        self, 
        layer_sizes: List[int], 
        neuron_params: dict, 
        synaptic_rule: BioLearningRule, # learning_rule -> synaptic_rule に名前変更
        homeostatic_rule: Optional[BioLearningRule] = None, # 安定化ルールを追加
        sparsification_config: Optional[Dict[str, Any]] = None
    ):
    # --- ▲ 改善 (v2) ▲ ---
        super().__init__()
        self.layer_sizes = layer_sizes
        # --- ▼ 改善 (v2): 2種類のルールを保持 ▼ ---
        self.synaptic_rule = synaptic_rule
        self.homeostatic_rule = homeostatic_rule
        # --- ▲ 改善 (v2) ▲ ---
        self.sparsification_enabled = sparsification_config.get("enabled", False) if sparsification_config else False
        self.contribution_threshold = sparsification_config.get("contribution_threshold", 0.0) if sparsification_config else 0.0
        if self.sparsification_enabled:
            print(f"🧬 適応的因果スパース化が有効です (貢献度閾値: {self.contribution_threshold})")
        if self.homeostatic_rule:
            print(f"⚖️ 恒常性維持ルール ({type(self.homeostatic_rule).__name__}) が有効です。")

        self.layers = nn.ModuleList()
        self.weights = nn.ParameterList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(BioLIFNeuron(layer_sizes[i+1], neuron_params))
            weight = nn.Parameter(torch.rand(layer_sizes[i+1], layer_sizes[i]) * 0.5)
            self.weights.append(weight)

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_spikes_history = []
        current_spikes = input_spikes
        for i, layer in enumerate(self.layers):
            current = torch.matmul(self.weights[i], current_spikes)
            current_spikes = layer(current)
            hidden_spikes_history.append(current_spikes)
        return current_spikes, hidden_spikes_history


    def update_weights(
        self,
        all_layer_spikes: List[torch.Tensor],
        optional_params: Optional[Dict[str, Any]] = None
    ):
        if not self.training:
            return

        backward_credit: Optional[torch.Tensor] = None
        current_params = optional_params.copy() if optional_params else {}

        for i in reversed(range(len(self.weights))):
            # 入力スパイクを取得（i=0の場合はall_layer_spikes[0]=input_spikesを使用）
            pre_spikes = all_layer_spikes[i]
            # 出力スパイクを取得（iに対応する層の出力は all_layer_spikes[i+1]）
            post_spikes = all_layer_spikes[i+1]


            if backward_credit is not None:
                # 階層的クレジット信号を適用
                reward_signal = current_params.get("reward", 0.0)
                # クレジット信号のスケール調整（例）
                modulated_reward = reward_signal + backward_credit.mean().item() * 0.1
                current_params["reward"] = modulated_reward
                # または causal_creditとして渡す
                # current_params["causal_credit"] = backward_credit.mean().item()

            # --- ▼ 改善 (v2): 2種類の学習則を適用 ▼ ---
            
            # 1. シナプス可塑性ルール (STDP, R-STDP, CausalTrace など)
            dw_synaptic, backward_credit_new = self.synaptic_rule.update(
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                weights=self.weights[i],
                optional_params=current_params
            )
            # 次のループのためにクレジット信号を更新
            backward_credit = backward_credit_new
            
            # 2. 恒常性維持ルール (BCM など)
            dw_homeostasis = torch.zeros_like(self.weights[i].data)
            if self.homeostatic_rule:
                # BCMなどは報酬信号を必要としないため、元の optional_params を渡す
                dw_homeo, _ = self.homeostatic_rule.update(
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes,
                    weights=self.weights[i],
                    optional_params=optional_params 
                )
                dw_homeostasis = dw_homeo

            # 最終的な重み変化量 = 可塑性 + 恒常性
            dw = dw_synaptic + dw_homeostasis
            # --- ▲ 改善 (v2) ▲ ---

            # --- ▼ 修正 ▼ ---
            # 適応的因果スパース化 (貢献度に基づく)
            # V2 クラス名に変更
            if self.sparsification_enabled and isinstance(self.synaptic_rule, CausalTraceCreditAssignmentEnhancedV2):
                # get_causal_contribution メソッドは V2 でも存在すると仮定
                causal_contribution = self.synaptic_rule.get_causal_contribution()
                if causal_contribution is not None:
                    # 貢献度が閾値以下の接続に対応する重み更新をゼロにする
                    contribution_mask = causal_contribution > self.contribution_threshold
                    dw = dw * contribution_mask
            # --- ▲ 修正 ▲ ---

            self.weights[i].data += dw
            self.weights[i].data.clamp_(min=0) # 例: 非負制約