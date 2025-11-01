# ファイルパス: scripts/observe_brain_thought_process.py
#
# Title: 思考の観察（人工脳との対話）
#
# Description:
# 統合されたArtificialBrainが、多様な感情的テキスト入力に対し、
# どのように感じ、記憶し、意思決定するのか、その「思考プロセス」を
# 詳細に観察するための対話型スクリプト。

import sys
from pathlib import Path
import time
import argparse

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import BrainContainer

def main():
    """
    DIコンテナを使って人工脳を初期化し、思考プロセスを観察しながら
    対話形式のシミュレーションを実行する。
    """
    parser = argparse.ArgumentParser(
        description="人工脳 思考プロセス観察ツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/small.yaml",
        help="モデルアーキテクチャ設定ファイルのパス。"
    )
    args = parser.parse_args()

    # 1. DIコンテナを初期化し、設定ファイルをロード
    container = BrainContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(args.model_config)

    # 2. コンテナから完成品の人工脳インスタンスを取得
    brain = container.artificial_brain()

    # 3. 対話ループの開始
    print("🧠 人工脳との対話を開始します。'exit' と入力すると終了します。")
    print("   喜び、怒り、悲しみなど感情豊かな文章や、複雑な質問を入力して、AIの思考を探ってみましょう。")
    print("-" * 70)

    try:
        while True:
            # ユーザーからの入力を受け付け
            input_text = input("あなた: ")
            if input_text.lower() == 'exit':
                break
            if not input_text:
                continue

            # --- 認知サイクルの実行 ---
            brain.run_cognitive_cycle(input_text)

            # --- 思考プロセスの観察 ---
            print("\n" + "="*20 + " 🔍 思考プロセスの観察 " + "="*20)
            
            # 感情 (Amygdala) の状態を表示
            emotion_state = brain.global_context.get('internal_state', {}).get('emotion', {})
            print(f"感情評価 (Amygdala): Valence={emotion_state.get('valence', 0.0):.2f}, Arousal={emotion_state.get('arousal', 0.0):.2f}")

            # 意思決定 (Basal Ganglia) の状態を表示 (直近のログから類推)
            # basal_ganglia.select_action 内のprint出力を観察します。
            
            # 記憶 (Hippocampus & Cortex) の状態を表示
            recent_memory = brain.hippocampus.retrieve_recent_episodes(1)
            print(f"短期記憶 (Hippocampus): 直近のエピソードを保持 - {len(brain.hippocampus.working_memory)}件")
            
            # 5サイクルごとに記憶が固定化される様子を観察
            if brain.cycle_count % 5 == 0:
                print("長期記憶 (Cortex): 記憶の固定化が実行されました。現在の知識グラフ:")
                # Cortexに保存されている知識の一部を表示
                all_knowledge = brain.cortex.get_all_knowledge()
                if all_knowledge:
                    # 最初の5件のコンセプトと関連情報を表示
                    for i, (concept, relations) in enumerate(all_knowledge.items()):
                        if i >= 5:
                            print("  ...")
                            break
                        print(f"  - Concept '{concept}': {relations}")
                else:
                    print("  - (まだ知識はありません)")

            print("="*64 + "\n")


    except KeyboardInterrupt:
        print("\n👋 対話ループを終了しました。")
    finally:
        print("\n長期記憶 (Cortex) に蓄積された最終的な知識グラフ:")
        all_knowledge = brain.cortex.get_all_knowledge()
        if all_knowledge:
            import json
            print(json.dumps(all_knowledge, indent=2, ensure_ascii=False))
        else:
            print("  (知識は蓄積されませんでした)")


if __name__ == "__main__":
    main()