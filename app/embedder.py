"""
向量化模块（Embedding）—— 把文本转成向量。

设计要点（面试可讲）：
- Embedding 把「语义相近的文本」映射到向量空间里「距离相近的位置」，
  于是"语义相似"就变成了可计算的向量距离。
- 用 fastembed（ONNX Runtime 推理）：免装 torch、轻量、快，
  且 ONNX 是端侧部署的标准格式，正好契合后续部署到树莓派的定位。
- 针对中文知识库选 BAAI/bge-small-zh-v1.5（中文语义效果优于通用多语言模型）。
"""
from pathlib import Path

from fastembed import TextEmbedding

# 本地 ONNX 模型（从 modelscope 下载，避免依赖 HF 网络）
_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "bge-small-zh-v1.5")
_model = None


def _get_model():
    global _model
    if _model is None:
        # specific_model_path 指向本地已下载的 ONNX 模型，避免联网下载
        _model = TextEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            specific_model_path=_MODEL_PATH,
        )
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """把文本列表转成向量列表。"""
    if isinstance(texts, str):
        texts = [texts]
    model = _get_model()
    return [v.tolist() for v in model.embed(texts)]


def embed_query(text: str) -> list[float]:
    """单条查询文本转成向量。"""
    return embed([text])[0]
