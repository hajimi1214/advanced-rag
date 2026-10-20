"""
FastAPI 入口 —— 提供 HTTP API，串起「建库 → 检索 → 生成」完整链路。

路由：
- POST /ingest   上传文档入库
- GET  /stats    查看库中 chunk 数
- POST /search   混合检索（不生成答案）
- POST /ask      完整问答（混合检索 → 重排序 → LLM 生成带引用答案）
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .hybrid import HybridRetriever
from .llm import generate_answer
from .pipeline import ingest_file
from .rerank import rerank
from .store import VectorStore

app = FastAPI(title="Advanced RAG 智能知识库问答系统", version="1.0.0")

DATA_DIR = Path("./data/docs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

store = VectorStore(persist_dir="./data/chroma", collection_name="knowledge_base")
retriever = HybridRetriever(store)


class Query(BaseModel):
    question: str
    top_k: int = 5


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """上传文档（pdf/docx/md/txt）并入库。"""
    path = DATA_DIR / file.filename
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    info = ingest_file(str(path), store)
    retriever._build_bm25()  # 新文档入库后重建 BM25 索引
    return {"ok": True, **info}


@app.get("/stats")
def stats():
    return {"total_chunks": store.count()}


@app.post("/search")
def search(q: Query):
    """混合检索（向量 + BM25 + RRF），不生成答案。"""
    results = retriever.hybrid_search(q.question, top_k=q.top_k)
    return {"question": q.question, "results": results}


@app.post("/ask")
def ask(q: Query):
    """完整问答：混合检索粗召回 → 重排序精排 → LLM 生成带引用答案。"""
    candidates = retriever.hybrid_search(q.question, top_k=20)
    docs = [c["text"] for c in candidates]

    ranked = rerank(q.question, docs, top_k=q.top_k)

    contexts = []
    for doc, score in ranked:
        source = next((c["source"] for c in candidates if c["text"] == doc), "")
        contexts.append({"text": doc, "source": source, "score": round(score, 4)})

    # 优雅降级：LLM 未配置（无 API key 且无本地 Ollama）时，返回纯检索结果
    try:
        answer = generate_answer(q.question, contexts)
        mode = "llm"
    except Exception:
        answer = None
        mode = "retrieval_only"
    return {"question": q.question, "answer": answer, "contexts": contexts, "mode": mode}


# 前端界面（挂在最后，避免拦截上方 API 路由）
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
