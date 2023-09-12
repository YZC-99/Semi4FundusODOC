import torch
import torch.nn as nn

# 准备数据，假设你有一个形状为(256, 32)的Tensor作为输入数据
input_data = torch.ones(256, 32)

# 选择泰勒展开的阶数
order = 2

# 定义泰勒展开模型
class TaylorExpansionModel(nn.Module):
    def __init__(self, order):
        super(TaylorExpansionModel, self).__init__()
        self.order = order

    def forward(self, x):
        # 泰勒展开的多项式计算
        result = x.clone()  # 阶数为0的项
        for i in range(1, self.order + 1):
            term = torch.pow(x, i) / i  # 高阶项
            result += term
        return result

# 创建模型实例
model = TaylorExpansionModel(order)

# 使用模型进行建模
output = model(input_data)

# 输出结果
print(input_data)
print(output)
print(input_data-output)
