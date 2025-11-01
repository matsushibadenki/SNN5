# ファイルパス: snn_research/cognitive_architecture/hippocampus.py
# (更新)
#
# Title: Hippocampus (海馬) モジュール
#
# Description:
# - 人工脳アーキテクチャの「記憶層」に属し、短期記憶（ワーキングメモリ）を担う。
# - 新しい情報や経験を「エピソード」として時系列で短期的に保持する。
# - 保持できる情報量には限りがあり、古い記憶は忘却される（FIFO）。
# - 将来的には、長期記憶への転送（記憶の固定）や、
#   注意機構と連携した情報の重み付けなどの機能拡張を想定。
#
# 改善点(v2):
# - ROADMAPフェーズ3に基づき、長期記憶への固定化プロセスを明確にするためのメソッドを追加。
# 改善点(v3):
# - 「意識的認知サイクル」実装のため、GlobalWorkspaceと連携。
# - 新しい情報と短期記憶の関連性を評価し、その関連度を顕著性スコアとしてWorkspaceにアップロードする。

from typing import List, Dict, Any
from collections import deque
import torch

from .global_workspace import GlobalWorkspace

class Hippocampus:
    """
    短期的なエピソード記憶を管理し、記憶との関連性を評価する海馬モジュール。
    """
    def __init__(self, workspace: GlobalWorkspace, capacity: int = 100):
        """
        Args:
            workspace (GlobalWorkspace): 情報をアップロードするための中央ハブ。
            capacity (int): ワーキングメモリが保持できるエピソードの最大数。
        """
        self.workspace = workspace
        self.capacity = capacity
        self.working_memory: deque = deque(maxlen=capacity)
        print(f"🧠 海馬（ワーキングメモリ）モジュールが初期化されました (容量: {capacity} エピソード)。")

    def evaluate_relevance_and_upload(self, perception_features: torch.Tensor):
        """
        新しい知覚情報と短期記憶との関連性を評価し、結果をGlobalWorkspaceにアップロードする。
        """
        if not self.working_memory:
            salience = 0.8  # 記憶がなければ、新しい情報は常に顕著
            relevance_info = {"type": "memory_relevance", "relevance": 0.0, "details": "No existing memories."}
        else:
            # 簡易的な関連性評価：直近の記憶の特徴ベクトルとのコサイン類似度
            recent_episode = self.retrieve_recent_episodes(1)[0]
            recent_features = recent_episode.get('content', {}).get('features')
            
            if recent_features is not None and isinstance(recent_features, torch.Tensor):
                # ベクトルをフラット化してコサイン類似度を計算
                similarity = torch.nn.functional.cosine_similarity(
                    perception_features.flatten(), 
                    recent_features.flatten(), 
                    dim=0
                ).item()
                # 類似度が低い（新規性が高い）ほど顕著性が高い
                salience = 1.0 - similarity
                relevance_info = {"type": "memory_relevance", "relevance": similarity}
            else:
                salience = 0.7 # 比較対象がない場合
                relevance_info = {"type": "memory_relevance", "relevance": 0.0, "details": "Previous memory has no features."}

        self.workspace.upload_to_workspace(
            source="hippocampus",
            data=relevance_info,
            salience=salience
        )

    def store_episode(self, episode: Dict[str, Any]):
        """
        新しいエピソード（経験や観測）をワーキングメモリに保存する。
        """
        self.working_memory.append(episode)
        print(f"📝 海馬: 新しいエピソードを記憶しました。 (現在の記憶数: {len(self.working_memory)})")

    def retrieve_recent_episodes(self, num_episodes: int = 5) -> List[Dict[str, Any]]:
        """
        直近のいくつかのエピソードをワーキングメモリから検索して返す。
        """
        if num_episodes <= 0:
            return []
        num_to_retrieve = min(num_episodes, len(self.working_memory))
        return [self.working_memory[-i] for i in range(1, num_to_retrieve + 1)]
    
    def get_and_clear_episodes_for_consolidation(self) -> List[Dict[str, Any]]:
        """
        長期記憶への固定化のために、現在の全エピソードを返し、メモリをクリアする。
        """
        episodes_to_consolidate = list(self.working_memory)
        self.clear_memory()
        print(f"📤 海馬: 長期記憶への固定化のため、{len(episodes_to_consolidate)}件のエピソードを転送しました。")
        return episodes_to_consolidate

    def clear_memory(self):
        """
        ワーキングメモリの内容をすべて消去する。
        """
        self.working_memory.clear()
        print("🗑️ 海馬: ワーキングメモリをクリアしました。")