"""
LLM 生成模块 —— 基于检索结果生成带来源引用的答案。

支持两种后端（自动切换，面试可讲「可插拔 LLM 后端」）：
- 云端 OpenAI 兼容 API（DeepSeek / 通义 / OpenAI 等，通过环境变量配置）
- 本地 Ollama（端侧部署时用，跑量化小模型，全程离线）
"""
import os

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def _ask_ollama(prompt: str, model: str = "qwen2.5:1.5b") -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _ask_openai(prompt: str, model: str) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("未配置 OPENAI_API_KEY，请设置环境变量或改用本地 Ollama")
    resp = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=180,
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
    """基于检索上下文生成答案。

    Args:
        question: 用户问题
        contexts: 检索到的片段列表，每项含 text/source
        backend: "ollama" / "openai" / "auto"（auto 优先探测本地 Ollama）
        model: 模型名，不传则用默认
    """
    prompt = _build_prompt(question, contexts)

    if backend == "auto":
        # 探测本地 Ollama 是否可用
        try:
            requests.get("http://localhost:11434/api/tags", timeout=1)
            backend = "ollama"
        except requests.RequestException:
            backend = "openai"

    if backend == "ollama":
        return _ask_ollama(prompt, model or "qwen2.5:1.5b")
    return _ask_openai(prompt, model or "deepseek-chat")
