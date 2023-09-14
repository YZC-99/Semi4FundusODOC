import math
import matplotlib.pyplot as plt

# 定义参数
init_lr = 0.001
total_iters = 100  # 总的训练迭代次数
warmup_iter = 10  # warmup阶段的迭代次数
lr_min = 0.00001    # 最小学习率
lr_max = 0.001     # 最大学习率

# 在创建学习率调度器时传入初始学习率和 warm-up 步数
def warmup_cosine(current_step, initial_lr, warmup_steps):
    if current_step < warmup_steps:
        # 在 warm-up 阶段使用线性学习率增加
        return initial_lr * (float(current_step) / float(max(1, warmup_steps)))
    else:
        # 在余弦退火阶段使用余弦退火函数
        progress = float(current_step - warmup_steps) / float(max(1, total_iters - warmup_steps))
        return initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# 在创建学习率调度器时传入初始学习率和 warm-up 步数

# 定义模拟学习率变化的函数
# # #
# def warmup_cosine(cur_iter):
#     if cur_iter < warmup_iter:
#         return cur_iter / warmup_iter
#     else:
#         return (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((cur_iter - warmup_iter) / (total_iters - warmup_iter) * math.pi))) / 0.1

# 计算每个迭代步骤的学习率
learning_rates = [warmup_cosine(iteration,init_lr, warmup_iter) for iteration in range(total_iters)]

# 绘制学习率变化图像
plt.plot(range(total_iters), learning_rates)
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)
plt.show()
