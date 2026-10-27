from app.embedder import embed_query
import numpy as np

# 第1步：三句话。A 已经给你了，B 和 C 你来填：
a = embed_query("我爱吃苹果")
b = embed_query("我爱吃水果")   # 填一句和 A 意思差不多的话
c = embed_query("你讨厌牛马手机")   # 填一句和 A 完全无关的话

# 第2步：余弦相似度公式（已经写好，你只需要看懂它在干嘛：
# 两个向量点乘，除以它俩的长度相乘——就是量"方向有多一致"）
def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# 第3步：打印对比
print("A 和 B（意思相近）的相似度:", cos_sim(a, b))
print("A 和 C（不相关）的相似度:", cos_sim(a, c))

