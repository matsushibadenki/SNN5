# ファイルパス: run_brain_simulation.py
# (修正)
# 修正: DIコンテナがモデルアーキテクチャ設定（small.yaml）を読み込むように修正し、
#       実行時エラーを解消。
# 改善(v2): コマンドラインから単一の入力を受け取れるようにargparseを導入。

import sys
from pathlib import Path
import time
import argparse

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent))

from app.containers import BrainContainer

def main():
    """
    DIコンテナを使って人工脳を初期化し、シミュレーションを実行する。
    """
    parser = argparse.ArgumentParser(description="Artificial Brain Simulation Runner")
    parser.add_argument("--prompt", type=str, help="人工脳への単一の入力テキスト。指定しない場合はデモを実行します。")
    args = parser.parse_args()

    # 1. DIコンテナを初期化し、設定ファイルをロード
    container = BrainContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml("configs/models/small.yaml") # モデル設定を追加

    # 2. コンテナから完成品の人工脳インスタンスを取得
    brain = container.artificial_brain()

    # 3. シミュレーションの実行
    if args.prompt:
        # 単一の入力で実行
        print(f"--- Running single cognitive cycle for input: '{args.prompt}' ---")
        brain.run_cognitive_cycle(args.prompt)
    else:
        # デモモード
        print("--- Running demonstration with multiple inputs ---")
        inputs = [
            "素晴らしい発見だ！これは成功に繋がるだろう。",
            "エラーが発生しました。システムに問題があるようです。",
            "今日は穏やかな一日だ。"
        ]
        for text_input in inputs:
            brain.run_cognitive_cycle(text_input)
            time.sleep(1) # 各サイクルの間に少し待機

if __name__ == "__main__":
    main()