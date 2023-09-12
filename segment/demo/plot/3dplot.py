import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

# 创建一个示例的三维张量，这里使用随机数据
tensor = np.random.rand(10, 10, 10)

# 获取张量的形状
x, y, z = tensor.shape

# 创建三维热力图
heatmap = go.Volume(
    z=tensor,
    opacity=0.8,  # 设置透明度
    colorscale='Viridis',  # 颜色映射
)

# 创建布局
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X轴'),
        yaxis=dict(title='Y轴'),
        zaxis=dict(title='Z轴')
    )
)

# 创建图表对象
fig = go.Figure(data=[heatmap], layout=layout)

# 显示图表
# pyo.plot(fig, filename='heatmap.html')
fig.show()
