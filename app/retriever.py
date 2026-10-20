"""
检索模块 —— 从向量库检索与查询最相关的文本块。

当前（M2）：基础向量检索（单一向量相似度）。
后续（M3）：升级为混合检索（向量 + BM25 关键词 + RRF 融合）。
"""
from .store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """返回 top_k 个最相关的块，每项含文本、元数据、相似度分数。

        说明：ChromaDB 返回的是 cosine 距离（越小越相似），
        这里转成相似度 score（越大越相似），方便后续 RRF 融合统一处理。
        """
        res = self.store.search(query, top_k)
        results = []
        for doc, meta, dist in zip(res["documents"], res["metadatas"], res["distances"]):
            results.append({
                "text": doc,
                "metadata": meta or {},
                "score": round(1 - dist, 4),  # 距离 → 相似度
            })
        return results
