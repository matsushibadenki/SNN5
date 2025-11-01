# ファイルパス: snn_research/cognitive_architecture/planner_snn.py
# Phase 3: 学習可能な階層的思考プランナーSNN
#
# 機能:
# - 自然言語のタスク要求を入力として受け取る。
# - 利用可能な専門家スキル（サブタスク）の最適な実行順序を予測して出力する。
# - BreakthroughSNNをベースアーキテクチャとして使用する。
#
# 修正点(v2):
# - mypyエラー[override]を解消するため、forwardメソッドのシグネチャを
#   親クラスと一致するように修正。
#
# 修正(v3): mypy [override] エラーを再度修正。
#           return_full_mems 引数をシグネチャに追加。

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

from snn_research.core.snn_core import BreakthroughSNN

class PlannerSNN(BreakthroughSNN):
    """
    タスク要求からサブタスクのシーケンスを生成することに特化したSNNモデル。
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, num_skills: int, neuron_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            num_skills (int): 予測対象となるスキル（サブタスク）の総数。
        """
        super().__init__(vocab_size, d_model, d_state, num_layers, time_steps, n_head, neuron_config=neuron_config)
        
        # BreakthroughSNNの出力層を、スキルを予測するための分類層に置き換える
        self.output_projection = nn.Linear(d_state * num_layers, num_skills)
        print(f"🧠 学習可能プランナーSNNが {num_skills} 個のスキルを認識して初期化されました。")

    # --- ▼ 修正 ▼ ---
    # 修正: return_full_mems 引数を追加
    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_spikes: bool = False,
        output_hidden_states: bool = False, 
        return_full_hiddens: bool = False,
        return_full_mems: bool = False, # <<-- ADDED
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # --- ▲ 修正 ▲ ---
        """
        フォワードパスを実行し、スキル予測ロジット、スパイク、膜電位を返す。
        """
        # PlannerSNNは常にスキルロジットを返すことを意図しているため、
        # super().forward()には output_hidden_states=False, return_full_hiddens=False を渡してロジットを取得する。
        # return_full_mems も False で固定
        skill_logits_over_time, spikes, mem = super().forward(
            input_ids, 
            return_spikes=return_spikes, 
            output_hidden_states=False,
            return_full_hiddens=False,
            return_full_mems=False, # 常にFalseを渡す
            **kwargs
        )
        
        # 最終タイムステップのロジットをプーリングして、最終的な計画予測とする
        final_skill_logits = skill_logits_over_time[:, -1, :]
        
        return final_skill_logits, spikes, mem
