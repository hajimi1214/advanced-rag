#!/bin/bash
# 启动智能知识库问答系统（FastAPI）
cd "$(dirname "$0")"
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
