# ファイルパス: snn_research/cognitive_architecture/basal_ganglia.py
# (更新)
# タイトル: 大脳基底核：情動変調を伴う行動選択モジュール
# 機能説明:
# - 脳の直接路（Go）と間接路（NoGo）の機能を模倣し、複数の選択肢から最適な行動を決定する。
# - Amygdalaから受け取った情動コンテキスト（快・不快、覚醒・沈静）に基づき、
#   意思決定の閾値を動的に調整する。例えば、危険を察知した場合（不快・高覚醒）、
#   より迅速に行動を起こせるように閾値を下げる。
# - 実行ログを強化し、情動が意思決定に与えた影響を明確に表示するようにした。
#
# 改善点(v2):
# - 「意識的認知サイクル」実装のため、GlobalWorkspaceと連携。
# - ブロードキャストされた「意識的情報」を購読し、行動選択のトリガーとする。

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

from .global_workspace import GlobalWorkspace

class BasalGanglia:
    """
    価値信号と情動文脈に基づいて行動選択を行う大脳基底核モジュール。
    """
    def __init__(self, workspace: GlobalWorkspace, selection_threshold: float = 0.5, inhibition_strength: float = 0.3):
        """
        Args:
            workspace (GlobalWorkspace): 情報を購読するための中央ハブ。
            selection_threshold (float): 行動を実行に移すための基本的な活性化レベル。
            inhibition_strength (float): 選択されなかった行動に対する抑制の強さ。
        """
        self.workspace = workspace
        self.base_threshold = selection_threshold
        self.inhibition_strength = inhibition_strength
        self.selected_action: Optional[Dict[str, Any]] = None
        
        # 意識的情報のコールバックとして自身を購読
        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🧠 大脳基底核モジュールが初期化され、Workspaceを購読しました。")

    def handle_conscious_broadcast(self, source: str, conscious_data: Dict[str, Any]):
        """
        GlobalWorkspaceからブロードキャストされた意識的情報を受け取り、行動選択のトリガーとする。
        """
        print(f"📬 大脳基底核: '{source}' からの意識的情報を受信。行動選択を評価します。")
        
        # ここではプランナーからの行動候補をダミーで生成
        # 将来的にPlannerがWorkspaceにアップロードしたものを利用する
        candidates = [
            {'action': 'investigate_perception', 'value': 0.8},
            {'action': 'reflect_on_emotion', 'value': 0.7},
            {'action': 'ignore', 'value': 0.3},
        ]
        
        emotion_context = conscious_data if conscious_data.get("type") == "emotion" else None
        
        self.select_action(candidates, emotion_context=emotion_context)

    def _modulate_threshold(self, emotion_context: Optional[Dict[str, float]]) -> float:
        """情動状態に基づいて行動選択の閾値を動的に調整する。"""
        if emotion_context is None:
            return self.base_threshold

        valence = emotion_context.get("valence", 0.0)
        arousal = emotion_context.get("arousal", 0.0)
        
        arousal_effect = -arousal * 0.2
        valence_effect = -valence * arousal * 0.1
        
        modulated_threshold = self.base_threshold + arousal_effect + valence_effect
        final_threshold = max(0.1, min(0.9, modulated_threshold))
        
        if final_threshold != self.base_threshold:
            print(f"  - 大脳基底核: 情動により閾値を調整 ({self.base_threshold:.2f} -> {final_threshold:.2f})")
        
        return final_threshold

    def select_action(
        self, 
        action_candidates: List[Dict[str, Any]],
        emotion_context: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        提示された行動候補の中から、実行すべき最適な行動を一つ選択する。
        """
        self.selected_action = None # 前回の選択をリセット
        if not action_candidates:
            print("🤔 大脳基底核: 行動候補が提示されませんでした。")
            return None
            
        current_threshold = self._modulate_threshold(emotion_context)

        values = torch.tensor([candidate.get('value', 0.0) for candidate in action_candidates])
        print(f"  - 大脳基底核: 検討中の行動候補: {[c.get('action') for c in action_candidates]}, 価値: {[round(v.item(), 2) for v in values]}")

        best_action_index = torch.argmax(values)
        best_action_value = values[best_action_index]

        if best_action_value >= current_threshold:
            self.selected_action = action_candidates[best_action_index]
            print(f"🏆 行動選択: '{self.selected_action.get('action')}' (活性値: {best_action_value:.2f}, 閾値: {current_threshold:.2f})")
            return self.selected_action
        else:
            print(f"🤔 行動棄却: どの行動も実行閾値 ({current_threshold:.2f}) に達しませんでした。(最大活性値: {best_action_value:.2f})")
            return None