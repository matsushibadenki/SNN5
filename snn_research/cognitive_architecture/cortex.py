# ファイルパス: snn_research/cognitive_architecture/cortex.py
# (更新)
# 修正: mypyエラー [annotation-unchecked] を解消するため、__init__に戻り値の型ヒントを追加。
# 改善(v2): ROADMAPフェーズ3に基づき、Hippocampusからの実際のエピソードを
#            解釈してナレッジグラフを構築するロジックを実装。

from typing import Dict, Any, Optional, List
import re

class Cortex:
    """
    長期的な知識をナレッジグラフとして管理する大脳皮質モジュール。
    """
    def __init__(self) -> None:
        # 知識を格納するためのグラフ構造 (辞書で簡易的に表現)
        # 例: {'concept_A': [{'relation': 'is_a', 'target': 'category_X'}]}
        self.knowledge_graph: Dict[str, List[Dict[str, Any]]] = {}
        print("🧠 大脳皮質（長期記憶）モジュールが初期化されました。")

    def consolidate_memory(self, episode: Dict[str, Any]) -> None:
        """
        短期記憶のエピソードを解釈し、長期記憶として知識グラフに統合（固定）する。

        Args:
            episode (Dict[str, Any]):
                Hippocampusから送られてきた単一の記憶エピソード。
        """
        source_input = episode.get("source_input")
        
        # 文字列の入力からキーワード（名詞や形容詞など）を簡易的に抽出
        if isinstance(source_input, str):
            # 5文字以上の単語をキーワードと見なす簡単なルール
            keywords = set(re.findall(r'\b[a-zA-Z]{5,}\b', source_input.lower()))
            
            if len(keywords) > 1:
                # 抽出されたキーワード間に「co-occurred_with」の関係を追加
                keyword_list = list(keywords)
                for i in range(len(keyword_list)):
                    for j in range(i + 1, len(keyword_list)):
                        self._add_relationship(keyword_list[i], "co-occurred_with", keyword_list[j])
                        self._add_relationship(keyword_list[j], "co-occurred_with", keyword_list[i])
                print(f"📚 大脳皮質: エピソードからキーワード間の関連性を学習しました: {keywords}")
            elif not keywords:
                 print("⚠️ 大脳皮質: 知識として統合するのに十分なキーワードが見つかりませんでした。")
        else:
            print("⚠️ 大脳皮質: 知識として統合するには情報が不十分なエピソードです。")

    def _add_relationship(self, source: str, relation: str, target: Any) -> None:
        """ナレッジグラフに関係性を追加する内部メソッド。"""
        if source not in self.knowledge_graph:
            self.knowledge_graph[source] = []
        
        # 重複する関係は追加しない
        if not any(r['relation'] == relation and r['target'] == target for r in self.knowledge_graph[source]):
            self.knowledge_graph[source].append({"relation": relation, "target": target})


    def retrieve_knowledge(self, concept: str) -> Optional[List[Dict[str, Any]]]:
        """
        指定された概念に関連する知識を長期記憶から検索する。

        Args:
            concept (str): 検索のキーとなる概念。

        Returns:
            Optional[List[Dict[str, Any]]]:
                見つかった関連知識のリスト。見つからない場合はNone。
        """
        print(f"🔍 大脳皮質: 概念 '{concept}' に関連する知識を検索中...")
        return self.knowledge_graph.get(concept)

    def get_all_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        現在保持している全ての知識グラフを返す。
        """
        return self.knowledge_graph