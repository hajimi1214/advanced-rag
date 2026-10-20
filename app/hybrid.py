"""
混合检索模块 —— 向量检索 + BM25 关键词检索 + RRF 融合。

设计要点（面试可讲）：
- 纯向量检索盲区：专有名词、精确 ID、罕见词（语义"像"但没精确命中）。
- 纯 BM25 盲区：不懂语义，只会关键词匹配，换个说法就搜不到。
- 混合检索 = 两者互补，覆盖「语义相似」和「精确匹配」两类需求。
- RRF（Reciprocal Rank Fusion）：按「排名」融合多路结果，无需手动调权重，
  对排序鲁棒，公式 score(d) = Σ 1/(k + rank)，k 常取 60。
"""
import jieba
from rank_bm25 import BM25Okapi

from .store import VectorStore


def _tokenize(text: str) -> list[str]:
    """中文用 jieba 分词，英文/数字按空白切。"""
    return [t for t in jieba.cut(text) if t.strip()]


def rrf_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """RRF 融合多路排序结果（每路是已排好序的 id 列表）。

    Returns:
        按融合分降序的 [(id, score)] 列表。
    """
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._build_bm25()

    def _build_bm25(self):
        """从向量库导出全部 chunk，构建 BM25 索引。"""
        self._ids, self._docs = self.store.get_all()
        if self._docs:
            tokenized = [_tokenize(d) for d in self._docs]
            self._bm25 = BM25Okapi(tokenized)

    def _vector_search(self, query: str, top_k: int) -> list[str]:
        """向量召回，返回按相似度排序的 id 列表。"""
        res = self.store.search(query, top_k)
        return res["ids"]

    def _bm25_search(self, query: str, top_k: int) -> list[str]:
        """BM25 召回，返回按相关度排序的 id 列表。"""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self._ids[i] for i in ranked]

    def hybrid_search(self, query: str, top_k: int = 5, recall_k: int = 20) -> list[dict]:
        """混合检索：向量 + BM25 各召回 recall_k 个，RRF 融合后取 top_k。

        Returns:
            [{"id", "text", "source", "rrf_score"}]
        """
        vec_ids = self._vector_search(query, recall_k)
        bm25_ids = self._bm25_search(query, recall_k)

        # RRF 融合
        fused = rrf_fusion([vec_ids, bm25_ids])
        top_ids = [doc_id for doc_id, _ in fused[:top_k]]

        # 组装结果
        id2doc = dict(zip(self._ids, self._docs))
        # 从 store 拿元数据（source 等）
        results = []
        for doc_id in top_ids:
            meta = self.store.collection.get(ids=[doc_id], include=["metadatas"])
            source = ""
            if meta.get("metadatas") and meta["metadatas"][0]:
                source = meta["metadatas"][0].get("source", "")
            results.append({
                "id": doc_id,
                "text": id2doc.get(doc_id, ""),
                "source": source,
                "rrf_score": round(dict(fused)[doc_id], 6),
            })
        return results
