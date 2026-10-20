# Advanced RAG — 智能知识库问答系统

一个**进阶 RAG**（检索增强生成）智能知识库问答系统：上传 PDF / Word / Markdown 文档，用自然语言提问，系统检索相关片段并生成**带来源引用**的答案。

## 亮点（简历技术点）

- **混合检索**：向量检索 + BM25 关键词检索，双路召回互补
- **RRF 融合**：按排名融合多路结果，无需手动调权重
- **重排序**：cross-encoder 对粗召回结果精排，提升准确率
- **答案溯源**：每个答案标注来源文档，天然抗幻觉
- **可插拔 LLM**：支持云端 API / 本地 Ollama（端侧部署）

## 技术栈

Python · FastAPI · ChromaDB · sentence-transformers · BM25 · jieba · PyMuPDF

## 快速开始

```bash
# 1. 创建虚拟环境（Python 3.10+）
python3.12 -m venv venv
source venv/bin/activate

# 2. 安装依赖（国内用户建议设置镜像）
pip install -r requirements.txt

# 3. 启动服务
uvicorn app.main:app --reload
```

然后打开 `http://localhost:8000/docs` 查看 API 文档。

## 使用流程

```bash
# 上传文档入库
curl -F "file=@./data/docs/raspberry_pi.txt" http://localhost:8000/ingest

# 问答
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "树莓派是什么？"}'
```

## 完整方案

详见 [PROJECT.md](./PROJECT.md)，包含技术架构、核心设计、面试要点、端侧（树莓派）部署方案。

## License

MIT
