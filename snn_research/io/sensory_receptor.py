# ファイルパス: snn_research/io/sensory_receptor.py
# (新規作成)
#
# Title: Sensory Receptor (感覚受容器)
#
# Description:
# - 人工脳アーキテクチャの「入力層」を担うコンポーネント。
# - 外部環境からの多様な感覚情報（テキスト、数値データなど）を受け取る。
# - 受け取った情報を、後続のSpikeEncoderが処理できるような
#   標準化された内部形式に変換する。

from typing import Dict, Any, Union

class SensoryReceptor:
    """
    外部からの感覚情報を受け取り、内部表現に変換するモジュール。
    """
    def __init__(self):
        print("👁️ 感覚受容器モジュールが初期化されました。")

    def receive(self, data: Union[str, float, Dict[str, Any]]) -> Dict[str, Any]:
        """
        外部からのデータを受け取り、標準化された辞書形式で返す。

        Args:
            data (Union[str, float, Dict[str, Any]]):
                入力される感覚データ。テキスト、数値、または辞書形式。

        Returns:
            Dict[str, Any]: 標準化された感覚情報。
                            例: {'type': 'text', 'content': 'hello'}
        """
        data_type = "unknown"
        content = data

        if isinstance(data, str):
            data_type = "text"
        elif isinstance(data, (int, float)):
            data_type = "numeric"
        elif isinstance(data, dict):
            # 辞書の場合は、そのままcontentとし、typeを明示
            data_type = data.get("type", "dict")
            content = data.get("content", data)

        print(f"📬 感覚受容器: '{data_type}' タイプの情報を受信しました。")
        return {"type": data_type, "content": content}