"""
文本分块模块 —— 把长文本切成适合向量化的片段。

设计要点（面试可讲）：
- 为什么要分块：LLM 上下文窗口有限，且检索粒度过大（整篇文档）会稀释语义，导致检索不准。
- chunk_size 怎么选：太小丢上下文、太大检索不精确；一般 300~800 字符，默认 500。
- chunk_overlap 的作用：在块之间保留重叠，避免一句话被切断导致语义断裂、漏检索。
- 分块方式：先按句号/换行切「句子单元」，再贪心合并，尽量在句子边界断开（比粗暴按字符切质量更高）。
"""
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def _split_sentences(text: str) -> list[str]:
    """按句子边界（中文/英文标点 + 换行）切分，保留分隔符。"""
    parts = _SENTENCE_SPLIT.split(text)
    return [p for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """把文本切成 chunk 列表。

    Args:
        text: 原始文本
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠的字符数（用于保持上下文连续）
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += s
        else:
            if current.strip():
                chunks.append(current.strip())
            # 保留重叠：取当前块末尾 overlap 长度的字符，拼到下一块开头
            if chunk_overlap > 0 and len(current) > chunk_overlap:
                current = current[-chunk_overlap:] + s
            else:
                current = s

    if current.strip():
        chunks.append(current.strip())
    return chunks
