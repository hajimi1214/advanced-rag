"""
向量库模块 —— 用 ChromaDB 存储向量并做相似度检索。

设计要点（面试可讲）：
- ChromaDB 是轻量向量数据库，Python 友好、可持久化到本地目录，适合本地/端侧场景
  （对比 Milvus/Weaviate 更重、需要单独部署）。
- 用 cosine 距离度量语义相似度。
- 「存储」与「检索」分离：M3 的混合检索会在此基础上扩展。
"""
import uuid

import chromadb

from .embedder import embed, embed_query


class VectorStore:
    def __init__(self, persist_dir: str = "./data/chroma", collection_name: str = "knowledge_base"):
        # 持久化到本地，重启不丢数据
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """把文本块向量化后写入向量库，返回 chunk id 列表。"""
        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = embed(chunks)
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    def search(self, query: str, top_k: int = 5) -> dict:
        """基础向量检索：返回最相似的 top_k 个块及其距离、元数据。"""
        q_emb = embed_query(query)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return {
            "ids": res["ids"][0] if res["ids"] else [],
            "documents": res["documents"][0] if res["documents"] else [],
            "metadatas": res["metadatas"][0] if res["metadatas"] else [],
            "distances": res["distances"][0] if res["distances"] else [],
        }

    def get_all(self) -> tuple[list[str], list[str]]:
        """返回所有 chunk 的 (ids, documents)，用于构建 BM25 索引。"""
        res = self.collection.get(include=["documents"])
        return res["ids"], res["documents"] or []

    def count(self) -> int:
        return self.collection.count()
