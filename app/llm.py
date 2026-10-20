"""
LLM 生成模块 —— 基于检索结果生成带来源引用的答案。

后端探测顺序（backend=auto，面试可讲「可插拔 LLM 后端 + 优雅降级」）：
1. 本地 llama.cpp（OpenAI 兼容 API，默认 8900 端口）—— 数据不出设备、零成本
2. 本地 Ollama（11434）—— 端侧部署用
3. 云端 OpenAI 兼容 API（DeepSeek 等）—— 兜底
"""
import os

import requests

LLAMA_CPP_BASE = "http://localhost:8900/v1"
OLLAMA_URL = "http://localhost:11434/api/generate"


def _ask_llama_cpp(prompt: str, model: str = "qwen3-30b-a3b") -> str:
    """调用本地 llama.cpp 的 OpenAI 兼容接口。"""
    resp = requests.post(
        f"{LLAMA_CPP_BASE}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _ask_ollama(prompt: str, model: str = "qwen2.5:1.5b") -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _ask_cloud(prompt: str, model: str) -> str:
    """云端 OpenAI 兼容 API（DeepSeek / 通义等），需要环境变量。"""
    base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("未配置 OPENAI_API_KEY")
    resp = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _build_prompt(question: str, contexts: list[dict]) -> str:
    """把检索到的片段拼进 prompt，并要求模型标注引用来源。"""
    ctx = "\n\n".join(
        f"[来源{i + 1}]\n{c['text']}" for i, c in enumerate(contexts)
    )
    return (
        "请严格根据以下资料回答问题；如果资料不足以回答，请明确说「资料中未找到相关信息」，"
        "不要编造。回答时请标注引用的来源编号（如 [来源1]）。\n\n"
        f"资料：\n{ctx}\n\n"
        f"问题：{question}\n\n"
        "回答："
    )


def generate_answer(
    question: str,
    contexts: list[dict],
    backend: str = "auto",
    model: str | None = None,
) -> str:
    """基于检索上下文生成答案。"""
    prompt = _build_prompt(question, contexts)

    if backend == "auto":
        # ① 本地 llama.cpp：数据不出设备、零 API 成本
        try:
            requests.get(f"{LLAMA_CPP_BASE}/models", timeout=2)
            return _ask_llama_cpp(prompt, model or "qwen3-30b-a3b")
        except requests.RequestException:
            pass
        # ② 本地 Ollama
        try:
            requests.get("http://localhost:11434/api/tags", timeout=1)
            return _ask_ollama(prompt, model or "qwen2.5:1.5b")
        except requests.RequestException:
            pass
        # ③ 云端兜底
        return _ask_cloud(prompt, model or "deepseek-chat")

    if backend == "ollama":
        return _ask_ollama(prompt, model or "qwen2.5:1.5b")
    return _ask_cloud(prompt, model or "deepseek-chat")
