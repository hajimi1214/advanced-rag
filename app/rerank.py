"""
重排序模块（Reranking）—— 对粗召回结果做精排。

设计要点（面试可讲）：
- 完整版用 cross-encoder：把 query 和 doc 拼在一起逐字交互算相关性，
  精度高于双塔 embedding（向量检索），但更慢——所以用两阶段「粗召回 + 精排」。
- 当前为轻量占位：保持 RRF 融合后的顺序（输入已排序），接口与完整版一致，
  后续接 cross-encoder（如 bge-reranker）时，调用方无需改动。
"""


def rerank(query: str, docs: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """轻量重排序：当前保持输入顺序（输入已由 RRF 融合排序）。

    完整版替换方案：用 cross-encoder（如 fastembed 的 TextCrossEncoder
    加载 bge-reranker ONNX 模型）对 [query, doc] 逐对打分精排。
    """
    return [(doc, 1.0) for doc in docs[:top_k]]
