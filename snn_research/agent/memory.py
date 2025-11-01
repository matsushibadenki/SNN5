# ファイルパス: snn_research/agent/memory.py
# (更新)
# Title: 長期記憶システム
# 改善点:
# - retrieve_similar_experiencesを強化し、因果スナップショットに基づいた検索を可能にする。

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from snn_research.cognitive_architecture.rag_snn import RAGSystem

class Memory:
    """
    エージェントの経験を構造化されたタプルとして長期記憶に記録し、
    RAGSystemと連携してセマンティック検索および因果的検索を行うクラス。
    """
    def __init__(self, rag_system: RAGSystem, memory_path: Optional[str] = "runs/agent_memory.jsonl"):
        self.rag_system = rag_system
        if memory_path is None:
            self.memory_path: str = "runs/agent_memory.jsonl"
        else:
            self.memory_path = memory_path
        
        if os.path.dirname(self.memory_path):
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)

    def _experience_to_text(self, experience: Dict[str, Any]) -> str:
        # (変更なし)
        action = experience.get("action", "NoAction")
        result = experience.get("result", {})
        reward = experience.get("reward", {}).get("external", 0.0)
        reason = experience.get("decision_context", {}).get("reason", "NoReason")
        return f"Action '{action}' was taken because '{reason}', resulting in '{str(result)}' with a reward of {reward:.2f}."

    def record_experience(
        self,
        state: Dict[str, Any],
        action: str,
        result: Any,
        reward: Dict[str, Any],
        expert_used: List[str],
        decision_context: Dict[str, Any],
        causal_snapshot: Optional[str] = None
    ):
        # (変更なし)
        experience_tuple = {
            "timestamp": datetime.utcnow().isoformat(),
            "state": state,
            "action": action,
            "result": result,
            "reward": reward,
            "expert_used": expert_used,
            "decision_context": decision_context,
            "causal_snapshot": causal_snapshot,
        }
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(experience_tuple, ensure_ascii=False) + "\n")
        
        experience_text = self._experience_to_text(experience_tuple)
        self.rag_system.add_relationship(
            source_concept=f"experience_{experience_tuple['timestamp']}",
            relation="is_described_as",
            target_concept=experience_text
        )
        
    # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
    def retrieve_similar_experiences(
        self,
        query_state: Optional[Dict[str, Any]] = None,
        causal_query: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        現在の状態や因果関係に類似した過去の経験を検索する。
        因果クエリが与えられた場合、それを優先的に使用する。
        """
        if causal_query:
            print(f"🧠 因果的記憶を検索中: {causal_query}")
            # RAGSystemを使って因果関係が記録されたナレッジを検索
            search_results = self.rag_system.search(f"Find causal relation similar to: {causal_query}", k=top_k)
            # この例では、検索されたテキストそのものを返す
            return [{"retrieved_causal_text": res} for res in search_results]

        if query_state:
            # 従来の状態に基づくセマンティック検索
            query_text = f"Find similar past experiences for a situation where the last action was '{query_state.get('last_action')}' and the result was '{str(query_state.get('last_result'))}'."
            print(f"🧠 過去の経験を検索中: {query_text}")
            search_results = self.rag_system.search(query_text, k=top_k)
            return [{"retrieved_text": res} for res in search_results]

        return []
    # --- ◾️◾️◾️◾️◾️↑ここまでが重要↑◾️◾️◾️◾️◾️ ---

    def retrieve_successful_experiences(self, top_k: int = 5) -> List[Dict[str, Any]]:
        # (変更なし)
        experiences = []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                for line in f: experiences.append(json.loads(line))
        except FileNotFoundError: return []

        def get_total_reward(exp: Dict[str, Any]) -> float:
            reward_info = exp.get("reward", {})
            if isinstance(reward_info, dict):
                return float(reward_info.get("external", 0.0))
            elif isinstance(reward_info, (int, float)):
                return float(reward_info)
            return 0.0

        experiences.sort(key=get_total_reward, reverse=True)
        return experiences[:top_k]