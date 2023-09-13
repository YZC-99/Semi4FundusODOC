import math
import matplotlib.pyplot as plt

# 定义参数
total_iters = 1000  # 总的训练迭代次数
warmup_iter = 100  # warmup阶段的迭代次数
lr_min = 0.00001    # 最小学习率
lr_max = 0.0001     # 最大学习率

# 定义模拟学习率变化的函数
def warmup_cosine(cur_iter):
    if cur_iter < warmup_iter:
        return cur_iter / warmup_iter
    else:
        return (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((cur_iter - warmup_iter) / (total_iters - warmup_iter) * math.pi))) / 0.1

# 计算每个迭代步骤的学习率
learning_rates = [warmup_cosine(iteration) for iteration in range(total_iters)]

# 绘制学习率变化图像
plt.plot(range(total_iters), learning_rates)
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)
plt.show()
