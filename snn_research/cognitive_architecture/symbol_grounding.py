# ファイルパス: snn_research/cognitive_architecture/symbol_grounding.py
# (更新)
#
# Title: 記号創発システム (Symbol Grounding System)
#
# Description:
# - ROADMAPフェーズ7に基づき、ナレッジグラフとしての機能を追加。
# - 改善点 (v2): ロードマップ「ニューラル活動への記号接地」を実装。
#   SNN内部の安定した発火パターン（テンソル）を受け取り、
#   それに新しいシンボルを割り当てる`ground_neural_pattern`メソッドを追加。

from typing import Set, Dict, Any
import hashlib
import torch

from .rag_snn import RAGSystem

class SymbolGrounding:
    """
    観測から新しいシンボルを創発し、ナレッジグラフに定着させるシステム。
    """
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.known_concepts: Set[str] = set()
        self.concept_counter = 100 # concept_100から開始

    def _get_observation_hash(self, observation: Dict[str, Any]) -> str:
        """観測内容（辞書）から一意のハッシュを生成する"""
        s = str(sorted(observation.items()))
        return hashlib.sha256(s.encode()).hexdigest()

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def _get_pattern_hash(self, pattern: torch.Tensor) -> str:
        """ニューロン発火パターン（テンソル）から一意のハッシュを生成する"""
        # テンソルをバイト列に変換してハッシュ化
        pattern_bytes = pattern.cpu().numpy().tobytes()
        return hashlib.sha256(pattern_bytes).hexdigest()

    def ground_neural_pattern(self, pattern: torch.Tensor, context: str):
        """
        新しいニューロン発火パターンを処理し, 未知であれば新しいシンボルを割り当てる。

        Args:
            pattern (torch.Tensor): 観測されたニューロンの発火パターン。
            context (str): パターンが観測された文脈 (例: 'perception_layer_output')。
        """
        pattern_hash = self._get_pattern_hash(pattern)

        if pattern_hash not in self.known_concepts:
            self.known_concepts.add(pattern_hash)
            new_concept_id = f"neural_concept_{self.concept_counter}"
            self.concept_counter += 1

            print(f"✨ 新しいニューラル活動パターンを発見！ シンボル '{new_concept_id}' を割り当てます。")

            # ナレッジグラフへの記録
            self.rag_system.add_relationship(
                source_concept=new_concept_id,
                relation="is a",
                target_concept="stable neural pattern"
            )
            self.rag_system.add_relationship(
                source_concept=new_concept_id,
                relation="was observed in",
                target_concept=context
            )
            # パターンの統計情報も記録
            self.rag_system.add_relationship(
                source_concept=new_concept_id,
                relation="has activation level",
                target_concept=f"{pattern.float().mean().item():.4f}"
            )
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def process_observation(self, observation: Dict[str, Any], context: str):
        """
        新しい観測（辞書）を処理し、未知であれば新しいシンボルを割り当てる。
        """
        if not isinstance(observation, dict):
            return

        obs_hash = self._get_observation_hash(observation)

        if obs_hash not in self.known_concepts:
            self.known_concepts.add(obs_hash)
            new_concept_id = f"observation_concept_{self.concept_counter}"
            self.concept_counter += 1
            
            print(f"✨ 新しい外部観測を発見！ シンボル '{new_concept_id}' を割り当てます。")

            # ナレッジグラフへの記録
            self.rag_system.add_relationship(
                source_concept=new_concept_id,
                relation="was observed during",
                target_concept=context
            )
            for key, value in observation.items():
                if isinstance(value, (str, int, float)):
                    self.rag_system.add_relationship(
                        source_concept=new_concept_id,
                        relation=f"has property '{key}'",
                        target_concept=str(value)
                    )