# ファイルパス: snn_research/cognitive_architecture/prefrontal_cortex.py
# (更新)
#
# Title: 前頭前野（内発的動機主導）
#
# Description:
# - 高レベルの目標設定と戦略選択を行う前頭前野モジュール。
# - GlobalWorkspaceからの意識的情報と、IntrinsicMotivationSystemからの
#   内部状態（好奇心、退屈など）に基づいて意思決定を行う。
#
# 改善点(v3):
# - decide_goalメソッドを改修し、外部からの要求よりも先に、
#   AI自身の「好奇心」や「退屈」といった内発的動機に基づいて目標を決定するロジックを実装。

from typing import Dict, Any

from .global_workspace import GlobalWorkspace
from .intrinsic_motivation import IntrinsicMotivationSystem

class PrefrontalCortex:
    """
    高レベルの目標設定と戦略選択を行う前頭前野モジュール。
    GlobalWorkspaceからの意識的情報と内発的動機に基づいて意思決定を行う。
    """
    def __init__(self, workspace: GlobalWorkspace, motivation_system: IntrinsicMotivationSystem) -> None:
        self.workspace = workspace
        self.motivation_system = motivation_system
        self.current_goal: str = "Explore and learn"
        # 意識的情報のコールバックとして自身を購読
        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🧠 前頭前野（実行制御）モジュールが初期化され、Workspaceを購読しました。")

    def handle_conscious_broadcast(self, source: str, conscious_data: Dict[str, Any]):
        """
        GlobalWorkspaceからブロードキャストされた意識的情報と、現在の内発的動機に基づいて目標設定を行う。
        """
        print(f"📬 前頭前野: '{source}' からの意識的情報を受信しました。")
        
        # 最新の内発的動機を取得
        internal_state = self.motivation_system.get_internal_state()
        
        # 意識に上った情報と内発的動機を統合してコンテキストを構築
        system_context = {
            "conscious_content": conscious_data,
            "internal_state": internal_state,
            "external_request": conscious_data if source == "receptor" else None # 入力層からの情報は外部要求とみなす
        }
        
        self.decide_goal(system_context)

    def decide_goal(self, system_context: Dict[str, Any]) -> str:
        """
        内発的動機と外部からの要求を評価し、次の高レベルな目標を決定する。
        """
        print("🤔 前頭前野: 次の目標を思考中...")

        internal_state = system_context.get("internal_state", {})
        conscious_content = system_context.get("conscious_content", {})
        external_request_data = system_context.get("external_request")

        # --- 優先度1: 内発的動機 ---
        if internal_state.get("boredom", 0.0) > 0.7:
            self.current_goal = "Try a new skill to reduce boredom"
            print(f"🎯 新目標（内発的動機 - 退屈）: {self.current_goal}")
            return self.current_goal

        if internal_state.get("curiosity", 0.0) > 0.8:
            self.current_goal = "Explore a new topic to satisfy curiosity"
            print(f"🎯 新目標（内発的動機 - 好奇心）: {self.current_goal}")
            return self.current_goal
        
        # --- 優先度2: 外部からの要求 ---
        if external_request_data:
            request = ""
            if isinstance(external_request_data, dict) and external_request_data.get("type") == "text":
                request = external_request_data.get("content", "")
            elif isinstance(external_request_data, str):
                request = external_request_data
            
            if request:
                self.current_goal = f"Fulfill external request: {request}"
                print(f"🎯 新目標（外部要求）: {self.current_goal}")
                return self.current_goal

        # --- 優先度3: 強い情動反応 ---
        if conscious_content.get("type") == "emotion" and abs(conscious_content.get("valence", 0.0)) > 0.7:
            emotion_desc = "positive" if conscious_content.get("valence", 0.0) > 0 else "negative"
            self.current_goal = f"Respond to strong {emotion_desc} emotion"
            print(f"🎯 新目標（情動反応）: {self.current_goal}")
            return self.current_goal
            
        # --- デフォルトの目標 ---
        self.current_goal = "Organize and optimize existing knowledge"
        print(f"🎯 新目標（デフォルト）: {self.current_goal}")
        return self.current_goal