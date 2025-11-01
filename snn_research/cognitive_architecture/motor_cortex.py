# ファイルパス: snn_research/cognitive_architecture/motor_cortex.py
# (修正)
#
# Title: Motor Cortex (運動野) モジュール
#
# Description:
# - mypyエラーを解消するため、Optional型を明示的にインポート・使用するよう修正。
# - 人工脳アーキテクチャの「運動層」の最終出力を担うコンポーネント。
# - 小脳から受け取った一連の精密な運動コマンドを、
#   実際のアクチュエータを駆動するための具体的な出力信号に変換する。
# - これにより、抽象的な行動計画が物理的なアクションとして結実する。

from typing import List, Dict, Any, Optional

class MotorCortex:
    """
    運動コマンドシーケンスを具体的なアクチュエータ信号に変換する運動野モジュール。
    """
    actuators: List[str]

    def __init__(self, actuators: Optional[List[str]] = None):
        """
        Args:
            actuators (Optional[List[str]], optional):
                制御対象となるアクチュエータのリスト。
                例: ['joint1', 'joint2', 'gripper']
                指定されない場合は、汎用の'output'を使用する。
        """
        if actuators is None:
            self.actuators = ['output_alpha', 'output_beta']
        else:
            self.actuators = actuators
        print("🧠 運動野モジュールが初期化されました。")

    def execute_commands(self, motor_commands: List[Dict[str, Any]]) -> List[str]:
        """
        小脳から受け取ったコマンドシーケンスを解釈し、実行ログを生成する。
        実際のハードウェア制御では、このメソッドがアクチュエータを駆動する。

        Args:
            motor_commands (List[Dict[str, Any]]):
                Cerebellumによって生成されたタイムスタンプ付きのコマンドリスト。

        Returns:
            List[str]: 実行されたアクションのログ。
        """
        execution_log: List[str] = []
        if not motor_commands:
            return execution_log

        print("🦾 運動野: コマンドシーケンスの実行を開始...")

        for command_data in motor_commands:
            timestamp = command_data.get('timestamp')
            command = command_data.get('command')

            # ここでは、コマンドを解釈してログを生成するダミー実装を行う。
            # 将来的には、ここで実際のアクチュエータ制御APIを呼び出す。
            log_entry = f"[T={timestamp:.2f}s] コマンド '{command}' を実行 -> アクチュエータ '{self.actuators[0]}' を作動"
            print(f"  - {log_entry}")
            execution_log.append(log_entry)

        print("✅ 運動野: 全コマンドの実行が完了しました。")
        return execution_log