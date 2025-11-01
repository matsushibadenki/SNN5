# ファイルパス: snn_research/cognitive_architecture/rag_snn.py
#
# Phase 3: RAG-SNN (Retrieval-Augmented Generation) システム
#
# 改善点 (v2):
# - ROADMAPフェーズ3「因果ナレッジグラフ」に基づき、
#   `add_causal_relationship`メソッドを追加。
#   これにより、「A causes B」のような因果関係をより明確に表現できるようになる。

import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class RAGSystem:
    """
    外部知識（ドキュメント）と内部記憶（エージェントログ）を検索し、
    思考のための文脈を提供するRAGシステム。
    ナレッジグラフとしての機能も併せ持つ。
    """
    def __init__(self, vector_store_path: str = "runs/vector_store"):
        self.vector_store_path = vector_store_path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store: Optional[FAISS] = self._load_vector_store()

    def _load_vector_store(self) -> Optional[FAISS]:
        """ベクトルストアをディスクから読み込む。"""
        if os.path.exists(self.vector_store_path):
            print(f"📚 既存のベクトルストアをロード中: {self.vector_store_path}")
            return FAISS.load_local(self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
        return None

    def setup_vector_store(self, knowledge_dir: str = "doc", memory_file: str = "runs/agent_memory.jsonl"):
        """
        知識源からドキュメントを読み込み、ベクトルストアを構築・保存する。
        """
        print("🛠️ ベクトルストアの構築を開始します...")
        
        doc_loader = DirectoryLoader(knowledge_dir, glob="**/*.md", loader_cls=TextLoader, silent_errors=True)
        txt_loader = DirectoryLoader(knowledge_dir, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
        docs = doc_loader.load() + txt_loader.load()

        if os.path.exists(memory_file):
            memory_loader = TextLoader(memory_file)
            docs.extend(memory_loader.load())
        
        if not docs:
            print("⚠️ 知識源となるドキュメントが見つかりませんでした。")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        print(f"📄 {len(split_docs)}個のドキュメントチャンクをベクトル化しています...")
        self.vector_store = FAISS.from_documents(split_docs, self.embedding_model)
        
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        print(f"✅ ベクトルストアの構築が完了し、'{self.vector_store_path}' に保存しました。")

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        クエリに最も関連するドキュメントチャンクを検索する。
        """
        if self.vector_store is None:
            print("ベクトルストアがセットアップされていません。先に setup_vector_store() を実行してください。")
            self.setup_vector_store()
            if self.vector_store is None:
                 return ["エラー: ベクトルストアを構築できませんでした。"]

        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def add_relationship(self, source_concept: str, relation: str, target_concept: str):
        """
        概念間の関係性をナレッジグラフ（ベクトルストア）に追加する。
        """
        if self.vector_store is None:
            print("⚠️ ベクトルストアが初期化されていません。新規作成します。")
            self.vector_store = FAISS.from_texts([], self.embedding_model)

        relationship_text = f"Concept Relation: {source_concept} {relation} {target_concept}."
        
        doc = Document(page_content=relationship_text, metadata={"source": "internal_knowledge"})
        self.vector_store.add_documents([doc])
        
        self.vector_store.save_local(self.vector_store_path)
        print(f"📈 ナレッジグラフ更新: 「{relationship_text}」")

    # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
    def add_causal_relationship(self, cause: str, effect: str, condition: Optional[str] = None):
        """
        概念間の因果関係をナレッジグラフに追加する。

        Args:
            cause (str): 原因となったイベント記述。
            effect (str): 結果として生じたイベント記述。
            condition (Optional[str]): 因果関係が成立した文脈・状況。
        """
        if self.vector_store is None:
            # ライブでベクトルストアが存在しない場合、空のストアを初期化
            self.vector_store = FAISS.from_texts([], self.embedding_model)

        if condition:
            causal_text = f"Causal Relation: Under condition '{condition}', the event '{cause}' leads to the effect '{effect}'."
        else:
            causal_text = f"Causal Relation: The event '{cause}' directly leads to the effect '{effect}'."
        
        doc = Document(page_content=causal_text, metadata={"source": "causal_inference"})
        self.vector_store.add_documents([doc])
        self.vector_store.save_local(self.vector_store_path)
        print(f"🔗 因果関係を記録: 「{causal_text}」")
    # --- ◾️◾️◾️◾️◾️↑ここまでが重要↑◾️◾️◾️◾️◾️ ---