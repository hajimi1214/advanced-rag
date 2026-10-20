"""
文档解析模块 —— 把 PDF / Word / Markdown / TXT 统一转成纯文本。

设计要点（面试可讲）：
- 用「多格式适配器」模式：每种格式一个解析器，按扩展名分发。
- 选 PyMuPDF（fitz）而非 pdfplumber：它快、省内存，适合大批量文档。
- 解析出来的文本后续统一交给 chunker 分块，保持「解析」与「分块」职责分离。
"""
from pathlib import Path

import pymupdf  # PyMuPDF
import docx  # python-docx


def load_document(path: str) -> str:
    """根据文件扩展名分发到对应解析器，返回纯文本。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(p)
    if ext in (".docx", ".doc"):
        return _load_docx(p)
    if ext in (".md", ".markdown", ".txt"):
        return _load_text(p)
    raise ValueError(f"不支持的文件格式: {ext}（支持 pdf / docx / md / txt）")


def _load_pdf(path: Path) -> str:
    """PDF 解析：逐页提取文本，页之间用换行分隔。"""
    doc = pymupdf.open(path)
    pages = []
    try:
        for page in doc:
            pages.append(page.get_text())
    finally:
        doc.close()
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    """Word 解析：提取段落文本，跳过空段落。"""
    d = docx.Document(str(path))
    paras = [p.text for p in d.paragraphs if p.text.strip()]
    return "\n\n".join(paras)


def _load_text(path: Path) -> str:
    """纯文本 / Markdown：直接读取。"""
    return path.read_text(encoding="utf-8", errors="ignore")
