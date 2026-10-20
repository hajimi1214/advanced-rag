"""
数据入库管道 —— 串联「文档 → 解析 → 分块 → 向量化 → 入库」。

这是整个 RAG 的"离线建库"阶段（对应架构图里的阶段一）。
"""
from .loader import load_document
from .chunker import chunk_text
from .store import VectorStore


def ingest_file(
    path: str,
    store: VectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> dict:
    """解析一个文件，分块后写入向量库，返回入库统计。"""
    text = load_document(path)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return {"file": path, "chunks": 0, "ids": []}

    metadatas = [{"source": path} for _ in chunks]
    ids = store.add_chunks(chunks, metadatas)
    return {"file": path, "chunks": len(chunks), "ids": ids}
