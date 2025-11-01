# ファイルパス: snn_research/training/bio_trainer.py
# Title: 生物学的強化学習用トレーナー
# Description: 強化学習のパラダイムに合わせ、エージェントと環境を引数に取るように変更。
#              エピソードベースの学習ループ（行動選択 -> 環境作用 -> 学習）を実装。
# 修正点:
# - mypyエラーを解消するため、GridWorldEnvに対応。
# - 複数ステップからなるエピソードベースの学習ループに修正。
# - mypyエラー `Incompatible types in assignment` を解消するため、
#   `episode_reward` をfloatで初期化するように修正。

import torch
from tqdm import tqdm  # type: ignore
from typing import Dict

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.rl_env.grid_world import GridWorldEnv

class BioRLTrainer:
    """生物学的強化学習エージェントのためのトレーナー。"""
    def __init__(self, agent: ReinforcementLearnerAgent, env: GridWorldEnv):
        self.agent = agent
        self.env = env

    def train(self, num_episodes: int) -> Dict[str, float]:
        """強化学習の学習ループを実行する。"""
        progress_bar = tqdm(range(num_episodes))
        total_rewards = []

        for episode in progress_bar:
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                # 1. 行動選択
                action = self.agent.get_action(state)
                
                # 2. 環境との相互作用
                next_state, reward, done = self.env.step(action)
                
                # 3. 学習
                self.agent.learn(reward)
                
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)
            avg_reward = sum(total_rewards[-20:]) / len(total_rewards[-20:])
            
            progress_bar.set_description(f"Bio RL Training Episode {episode+1}/{num_episodes}")
            progress_bar.set_postfix({"Last Reward": f"{episode_reward:.2f}", "Avg Reward (last 20)": f"{avg_reward:.3f}"})

        final_avg_reward = sum(total_rewards) / num_episodes if num_episodes > 0 else 0.0
        print(f"Training finished. Final average reward: {final_avg_reward:.4f}")
        return {"final_average_reward": final_avg_reward}