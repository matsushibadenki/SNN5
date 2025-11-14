# ファイルパス: snn_research/cognitive_architecture/vector_store.py
# (新規作成)
# Title: VectorStore (FAISSラッパー)
# Description:
#   RAGSystem (rag_snn.py) で使用するためのベクトル検索ラッパー。
#   FAISSインデックスの管理、ドキュメントとメタデータの保存・読み込み、
#   ベクトル検索機能を提供します。

import faiss  # type: ignore[import-untyped]
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """
    FAISS (Facebook AI Similarity Search) をラップし、
    テキストとメタデータの永続化を管理するクラス。
    """
    def __init__(self, storage_path: str = "runs/vector_store"):
        """
        VectorStoreを初期化します。

        Args:
            storage_path (str): FAISSインデックスとメタデータを保存するディレクトリ。
        """
        self.storage_path = storage_path
        self.index_path = os.path.join(storage_path, "vector_index.faiss")
        self.metadata_path = os.path.join(storage_path, "metadata.pkl")
        
        # 保存先ディレクトリを作成
        os.makedirs(storage_path, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        
        self.dimension: Optional[int] = None

        self.load()

    def _initialize_index(self, dimension: int) -> None:
        """指定された次元で新しいFAISSインデックスを初期化します。"""
        if self.index is None:
            logger.info(f"Initializing new FAISS IndexFlatIP (dimension={dimension}).")
            # IndexFlatIP は内積 (Inner Product) を使用
            # 正規化されたベクトルでは、内積はコサイン類似度と等価
            self.index = faiss.IndexFlatIP(dimension)
            self.dimension = dimension

    def add(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """
        テキスト、エンベディング、メタデータをストアに追加します。

        Args:
            texts (List[str]): ドキュメントチャンクのリスト。
            embeddings (np.ndarray): 対応するエンベディング (N, D)。
            metadatas (List[Dict[str, Any]]): 対応するメタデータのリスト。
        """
        if embeddings.shape[0] == 0:
            logger.warning("VectorStore.add: Received 0 embeddings. Nothing to add.")
            return
            
        if self.index is None:
            self._initialize_index(embeddings.shape[1])
        
        if self.index is None:
             logger.error("VectorStore.add: Index is still None after initialization attempt.")
             return
             
        if embeddings.shape[1] != self.dimension:
            logger.error(f"VectorStore.add: Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")
            return

        # データをFAISSインデックスに追加
        self.index.add(embeddings.astype(np.float32))
        
        # メタデータとテキストを内部リストに追加
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        
        logger.info(f"Added {len(texts)} items to VectorStore. Total items: {len(self.documents)}")
        self.save()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        クエリエンベディングを使用して、類似したドキュメントを検索します。

        Args:
            query_embedding (np.ndarray): 検索クエリのエンベディング (1, D)。
            k (int): 取得する結果の数。

        Returns:
            List[Tuple[str, Dict[str, Any], float]]: (テキスト, メタデータ, スコア) のリスト。
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("VectorStore.search: Index is empty or not initialized.")
            return []
            
        if query_embedding.shape[1] != self.dimension:
            logger.error(f"VectorStore.search: Query dimension mismatch. Expected {self.dimension}, got {query_embedding.shape[1]}")
            return []

        # 検索するkの値を、インデックス内の合計数までに制限
        k = min(k, self.index.ntotal)
        
        # FAISSで検索 (D = 距離/スコア, I = インデックスID)
        D, I = self.index.search(query_embedding.astype(np.float32), k)
        
        results: List[Tuple[str, Dict[str, Any], float]] = []
        for i in range(I.shape[1]):
            idx = I[0, i]
            score = D[0, i]
            
            if idx < 0 or idx >= len(self.documents):
                logger.warning(f"VectorStore.search: Invalid index {idx} found in search results.")
                continue
                
            results.append((self.documents[idx], self.metadatas[idx], float(score)))
            
        return results

    def save(self) -> None:
        """
        現在のインデックスとメタデータをディスクに保存します。
        """
        if self.index is not None:
            try:
                faiss.write_index(self.index, self.index_path)
                
                # メタデータ（テキストと次元情報を含む）をpickleで保存
                metadata_payload = {
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                    "dimension": self.dimension
                }
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(metadata_payload, f)
                
                logger.debug(f"VectorStore saved to {self.storage_path}")
                
            except Exception as e:
                logger.error(f"Failed to save VectorStore: {e}", exc_info=True)
        else:
            logger.debug("VectorStore.save: Index is None, nothing to save.")

    def load(self) -> None:
        """
        ディスクからインデックスとメタデータを読み込みます。
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                # メタデータを先に読み込み（次元情報を取得するため）
                with open(self.metadata_path, 'rb') as f:
                    metadata_payload = pickle.load(f)
                
                self.documents = metadata_payload["documents"]
                self.metadatas = metadata_payload["metadatas"]
                self.dimension = metadata_payload["dimension"]
                
                # FAISSインデックスを読み込み
                self.index = faiss.read_index(self.index_path)
                
                logger.info(f"VectorStore loaded successfully. {len(self.documents)} items, dimension={self.dimension}.")
                
                if self.index.ntotal != len(self.documents):
                     logger.warning(
                         f"VectorStore mismatch: FAISS index has {self.index.ntotal} items, "
                         f"but metadata has {len(self.documents)} items. Re-indexing might be needed."
                     )

            except Exception as e:
                logger.error(f"Failed to load VectorStore from {self.storage_path}: {e}. Initializing empty store.", exc_info=True)
                self.clear() # 破損している可能性があるのでクリア
        else:
            logger.info(f"No existing VectorStore found at {self.storage_path}. Initializing empty store.")

    def clear(self) -> None:
        """
        インデックスとメタデータをクリアし、ディスク上のファイルも削除します。
        """
        self.index = None
        self.documents = []
        self.metadatas = []
        self.dimension = None
        
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            logger.info(f"VectorStore cleared and files removed from {self.storage_path}.")
        except Exception as e:
            logger.error(f"Error while clearing VectorStore files: {e}", exc_info=True)