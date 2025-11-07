# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# (更新)
# 改善点:
# - `learn`メソッドに`causal_credit`引数を追加。
# - この引数が渡された場合、通常の報酬よりも優先し、
#   より大きな学習率で重みを更新するロジックを実装。
#
# 改善 (v2):
# - doc/The-flow-of-brain-behavior.md との整合性を高めるため、
#   ハードコードされていた学習ルール (R-STDP) を削除。
# - __init__ が `synaptic_rule` と `homeostatic_rule` を
#   外部から（DIコンテナ経由で）受け取れるように修正。
# - これにより、snn_research/bio_models/simple_network.py (v2) の
#   安定化機構をエージェントが利用できるようになる。

import torch
# --- ▼ 改善 (v2): 必要な型ヒントを追加 ▼ ---
from typing import Dict, Any, List, Optional
# --- ▲ 改善 (v2) ▲ ---

from snn_research.bio_models.simple_network import BioSNN
# --- ▼ 改善 (v2): BioLearningRule をインポート ▼ ---
from snn_research.learning_rules.base_rule import BioLearningRule
# --- ▲ 改善 (v2) ▲ ---
from snn_research.communication import SpikeEncoderDecoder

class ReinforcementLearnerAgent:
    """
    BioSNNと報酬変調型STDPを用い、トップダウンの因果クレジット信号で学習が変調される強化学習エージェント。
    """
    # --- ▼ 改善 (v2): __init__ のシグネチャを変更 ▼ ---
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        device: str,
        synaptic_rule: BioLearningRule, # 外部から注入
        homeostatic_rule: Optional[BioLearningRule] = None # 外部から注入
    ):
    # --- ▲ 改善 (v2) ▲ ---
        self.device = device
        
        # --- ▼ 改善 (v2): ハードコードされた学習ルールを削除 ▼ ---
        # learning_rule = RewardModulatedSTDP(
        #     learning_rate=0.005, a_plus=1.0, a_minus=1.0,
        #     tau_trace=20.0, tau_eligibility=50.0
        # )
        # --- ▲ 改善 (v2) ▲ ---
        
        hidden_size = (input_size + output_size) * 2
        layer_sizes = [input_size, hidden_size, output_size]
        
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
            # --- ▼ 改善 (v2): 注入されたルールを使用 ▼ ---
            synaptic_rule=synaptic_rule,
            homeostatic_rule=homeostatic_rule
            # --- ▲ 改善 (v2) ▲ ---
        ).to(device)

        self.encoder = SpikeEncoderDecoder(num_neurons=input_size, time_steps=1)
        self.experience_buffer: List[List[torch.Tensor]] = []

    def get_action(self, state: torch.Tensor) -> int:
        """
        現在の状態から、モデルの推論によって単一の行動インデックスを決定する。
        """
        self.model.eval()
        with torch.no_grad():
            input_spikes = (torch.rand_like(state) < (state * 0.5 + 0.5)).float()
            output_spikes, hidden_spikes_history = self.model(input_spikes)
            self.experience_buffer.append([input_spikes] + hidden_spikes_history)
            action = torch.argmax(output_spikes).item()
            return int(action)

    # --- ▼ 修正 ▼ ---
    def learn(self, reward: float, causal_credit: float = 0.0):
        """
        受け取った報酬信号または因果的クレジット信号を用いて、モデルの重みを更新する。
        """
        if not self.experience_buffer:
            return

        self.model.train()
        
        # 因果的クレジット信号が与えられた場合、それを優先し、学習を増幅させる
        if causal_credit > 0:
            # クレジット信号は通常の報酬よりも強力な学習トリガーとする
            final_reward_signal = reward + causal_credit * 10.0 
            print(f"🧠 シナプス学習増強！ (Causal Credit: {causal_credit})")
        else:
            final_reward_signal = reward
            
        optional_params = {"reward": final_reward_signal}
        
        for step_spikes in self.experience_buffer:
            self.model.update_weights(
                all_layer_spikes=step_spikes,
                optional_params=optional_params
            )
        
        # エピソードが終了、または強力な学習イベントが発生したらバッファをクリア
        if reward != -0.05 or causal_credit > 0:
            self.experience_buffer = []
    # --- ▲ 修正 ▲ ---