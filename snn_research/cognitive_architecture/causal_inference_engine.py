# ファイルパス: snn_research/cognitive_architecture/causal_inference_engine.py
# (更新)
# 改善点:
# - 因果関係を推論した際、その情報を「因果的クレジット信号」として
#   GlobalWorkspaceにブロードキャストする機能を追加。

from typing import Dict, Any, Optional, Tuple
from collections import defaultdict

from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

class CausalInferenceEngine:
    """
    意識の連鎖を観察し、文脈依存の因果関係を推論して知識グラフを構築するエンジン。
    """
    def __init__(
        self,
        rag_system: RAGSystem,
        workspace: GlobalWorkspace,
        inference_threshold: int = 3
    ):
        self.rag_system = rag_system
        self.workspace = workspace
        self.inference_threshold = inference_threshold
        
        self.previous_conscious_info: Optional[Dict[str, Any]] = None
        self.previous_context: Optional[str] = None
        self.co_occurrence_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        
        self.just_inferred: bool = False
        
        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🔍 因果推論エンジンが初期化され、Workspaceを購読しました。")

    def reset_inference_flag(self):
        self.just_inferred = False

    def _get_event_description(self, conscious_data: Optional[Dict[str, Any]]) -> Optional[str]:
        if not conscious_data:
            return None
        event_type = conscious_data.get("type")
        if event_type == "emotion":
            valence = conscious_data.get("valence", 0.0)
            return "strong_negative_emotion" if valence < -0.5 else "strong_positive_emotion" if valence > 0.5 else None
        elif event_type == "perception":
            return "novel_perception"
        elif isinstance(conscious_data, str) and conscious_data.startswith("Fulfill external request"):
             return "external_request_received"
        elif isinstance(conscious_data, dict) and 'action' in conscious_data:
            return f"action_{conscious_data['action']}"
        return "general_observation"

    def _get_context_description(self) -> str:
        # この実装はダミーです。実際のPFCから情報を取得する必要があります。
        return "general_context"

    def handle_conscious_broadcast(self, source: str, conscious_data: Dict[str, Any]):
        """
        意識に上った情報の連鎖と、その時の文脈を観察し、因果関係を推論し、クレジット信号を生成する。
        """
        current_event = self._get_event_description(conscious_data)
        previous_event = self._get_event_description(self.previous_conscious_info)
        current_context = self._get_context_description()

        if previous_event and current_event and self.previous_context:
            event_tuple = (self.previous_context, previous_event, current_event)
            self.co_occurrence_counts[event_tuple] += 1
            
            count = self.co_occurrence_counts[event_tuple]
            print(f"  - 因果推論: イベント組観測 -> ({self.previous_context}, {previous_event}, {current_event}), 回数: {count}")

            if count == self.inference_threshold:
                print(f"  - 🔥 因果関係を推論・記録！")
                self.rag_system.add_causal_relationship(
                    cause=previous_event,
                    effect=current_event,
                    condition=self.previous_context
                )
                self.just_inferred = True
                
                # --- ▼ 修正 ▼ ---
                # 成功した因果関係（報酬が高いなど）を特定し、クレジット信号をブロードキャスト
                # ここでは簡略化のため、推論が成立したこと自体をポジティブなイベントと見なす
                if previous_event.startswith("action_"):
                    credit_data = {
                        "type": "causal_credit",
                        "target_action": previous_event, # 例: "action_web_research"
                        "credit": 1.0 # ポジティブなクレジット
                    }
                    print(f"  - 📢 因果的クレジット信号を生成: {credit_data}")
                    self.workspace.upload_to_workspace(
                        source="causal_engine",
                        data=credit_data,
                        salience=0.95 # 非常に高い顕著性を持たせる
                    )
                # --- ▲ 修正 ▲ ---
        
        self.previous_conscious_info = conscious_data
        self.previous_context = current_context