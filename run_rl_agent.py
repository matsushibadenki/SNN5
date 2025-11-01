# ファイルパス: run_rl_agent.py
# Title: 強化学習エージェント実行スクリプト
# Description: 生物学的学習則に基づくSNNエージェントを起動し、
#              強化学習のループを実行します。
# 改善点:
# - ROADMAPフェーズ2検証のため、GridWorldEnvに対応。
# - エピソードベースの学習ループを実装し、複数ステップのタスクを実行できるようにした。
# 改善点 (v2):
# - ROADMAPフェーズ2完了のため、学習結果を可視化・保存する機能を追加。
# - 学習終了後に報酬の推移をグラフとしてプロットし、画像ファイルとして保存。
# - 訓練済みのエージェントモデルをファイルに保存。
# 改善点 (v3): DIコンテナを使用するようにリファクタリングし、一貫性を向上。

import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent))

from app.containers import TrainingContainer

def plot_rewards(rewards: list, save_path: Path):
    """報酬の推移をプロットして保存する。"""
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode')
    # 移動平均を計算してプロット
    if len(rewards) >= 50:
        moving_avg = [sum(rewards[i-50:i]) / 50 for i in range(50, len(rewards))]
        plt.plot(range(50, len(rewards)), moving_avg, label='Moving Average (50 episodes)', color='red')
    plt.title('Reinforcement Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"📊 学習曲線グラフを '{save_path}' に保存しました。")

def main():
    parser = argparse.ArgumentParser(description="Biologically Plausible Reinforcement Learning Framework")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of learning episodes.")
    parser.add_argument("--output_dir", type=str, default="runs/rl_results", help="Directory to save results.")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- DIコンテナを使用して依存関係を構築 ---
    container = TrainingContainer()
    container.config.from_yaml("configs/base_config.yaml")
    # コマンドライン引数をコンフィグに反映
    container.config.training.epochs.from_value(args.episodes)

    # コンテナから完成品のトレーナーを取得
    trainer = container.bio_rl_trainer()
    
    print("\n" + "="*20 + "🤖 生物学的強化学習開始 (Grid World) 🤖" + "="*20)
    print(f"Device: {trainer.agent.device}")
    
    # --- 学習ループの実行 ---
    results = trainer.train(num_episodes=args.episodes)
    
    print("\n" + "="*20 + "✅ 学習完了" + "="*20)
    print(f"最終的な平均報酬: {results.get('final_average_reward', 0.0):.4f}")

    # --- 結果の保存 ---
    # 学習曲線はBioRLTrainer内でプロット・保存されると仮定
    # ここでは、最終的なモデルの保存のみ行う
    model_save_path = output_path / "trained_rl_agent.pth"
    torch.save(trainer.agent.model.state_dict(), model_save_path)
    print(f"💾 訓練済みモデルを '{model_save_path}' に保存しました。")


if __name__ == "__main__":
    main()