import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

# 两个二维向量
vector_a = np.array([3, 4])  # 以3、4为坐标的向量
vector_b = np.array([1, 2])  # 以1、2为坐标的向量

# 计算余弦相似度
similarity = cosine_similarity(vector_a, vector_b)
print("余弦相似度:", similarity)

# 绘制向量及点积
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='r', label='A')
plt.quiver(0, 0, vector_b[0], vector_b[1], angles='xy', scale_units='xy', scale=1, color='b', label='B')
plt.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='r')
plt.quiver(vector_b[0], vector_b[1], vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='g')
plt.text(0.7, 1.5, f"Cosine Similarity: {similarity:.2f}", color='g')
plt.xlim(-1, 10)
plt.ylim(-1, 10)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Cosine Similarity Visualization')
plt.grid()
plt.show()
