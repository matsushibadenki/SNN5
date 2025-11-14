# ファイルパス: snn_research/cognitive_architecture/rag_snn.py
# Title: RAG (Retrieval-Augmented Generation) SNN システム
# Description:
#   VectorStore (FAISS) を使用したRAGシステムの実装。
#   外部ドキュメントをベクトル化して保存・検索し、SNNのコンテキストを拡張する。
#
# --- 修正 (v1) ---
# ModuleNotFoundError: No module named 'langchain.text_splitter' を解消するため、
# langchain のバージョンアップに対応し、インポート元を 'langchain_text_splitters' に変更。

import faiss # type: ignore[import-untyped]
import numpy as np
import os
from typing import List, Optional, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
from .vector_store import VectorStore
import logging

# --- ▼▼▼ 【!!! HPO修正 (langchain v-up) !!!】 ▼▼▼ ---
# 修正前: from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- ▲▲▲ 【!!! HPO修正 !!!】 ▲▲▲ ---


logger = logging.getLogger(__name__)

class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) システム。
    FAISSを使用したVectorStoreを管理し、テキストのチャンク化、
    エンベディング、検索を行う。
    """
    def __init__(
        self,
        vector_store_path: str = "runs/vector_store",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        RAGSystemを初期化します。

        Args:
            vector_store_path (str): VectorStore (FAISSインデックスとドキュメント) の
                                     保存/読み込み先ディレクトリ。
            model_name (str): テキストエンベディングに使用するHuggingFaceモデル名。
            chunk_size (int): テキストを分割する際のチャンクサイズ。
            chunk_overlap (int): チャンク間のオーバーラップサイズ。
        """
        self.vector_store_path = vector_store_path
        self.vector_store = VectorStore(vector_store_path)
        
        # エンベディングモデルの初期化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # 評価モード
        
        # テキストスプリッターの初期化
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"RAGSystem initialized. VectorStore path: {vector_store_path}")

    def _mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        SentenceTransformersのためのMean Pooling処理。
        """
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        テキストのリストからエンベディング（ベクトル）を生成します。

        Args:
            texts (List[str]): エンベディングするテキストのリスト。

        Returns:
            np.ndarray: (N, D) 次元のエンベディング配列。
        """
        if not texts:
            return np.array([])
            
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # テキストをバッチ処理（ここでは簡易的に全量を一度に処理）
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean Pooling を実行
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # CPUに戻し、NumPy配列に変換
        embeddings_np: np.ndarray = sentence_embeddings.cpu().numpy()
        
        # L2正規化 (FAISSのIndexFlatIPは内積計算だが、正規化ベクトル同士の
        # 内積はコサイン類似度と等価であり、安定することが多いため)
        faiss.normalize_L2(embeddings_np)
        
        logger.info(f"Embeddings generated with shape: {embeddings_np.shape}")
        return embeddings_np

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        ドキュメントをチャンク化し、VectorStoreに追加します。

        Args:
            documents (List[str]): 追加するドキュメント（テキスト本文）のリスト。
            metadatas (Optional[List[Dict[str, Any]]]): 各ドキュメントに対応する
                                                       メタデータ（ソース元など）のリスト。
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return

        logger.info(f"Adding {len(documents)} documents to RAG system...")
        
        # 1. ドキュメントをチャンク化
        chunks = self.text_splitter.create_documents(documents, metadatas=metadatas)
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_metadatas = [chunk.metadata for chunk in chunks]
        
        if not chunk_texts:
            logger.warning("Text splitter resulted in 0 chunks.")
            return

        # 2. チャンクのエンベディングを生成
        embeddings = self.get_embeddings(chunk_texts)
        
        # 3. VectorStoreに追加
        self.vector_store.add(chunk_texts, embeddings, chunk_metadatas)
        logger.info(f"Added {len(chunk_texts)} chunks to VectorStore.")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        クエリに最も関連するドキュメントチャンクを検索します。

        Args:
            query (str): 検索クエリ。
            k (int): 取得するチャンク数。

        Returns:
            List[Tuple[str, Dict[str, Any], float]]: 
                (チャンクテキスト, メタデータ, 類似度スコア) のリスト。
        """
        if not query:
            return []
            
        logger.debug(f"Searching RAG system with query: '{query}' (k={k})")
        
        # 1. クエリのエンベディングを生成
        query_embedding = self.get_embeddings([query])
        
        if query_embedding.shape[0] == 0:
            logger.warning("Failed to generate query embedding.")
            return []
            
        # 2. VectorStoreで検索
        results = self.vector_store.search(query_embedding, k=k)
        
        logger.debug(f"RAG search found {len(results)} results.")
        return results

    def clear(self) -> None:
        """
        VectorStore内のすべてのデータをクリアします。
        """
        self.vector_store.clear()
        logger.info("RAG system (VectorStore) cleared.")
