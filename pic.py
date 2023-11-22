import matplotlib.pyplot as plt
import numpy as np

# 生成一些示例数据
np.random.seed(42)
data_points = 50

# 生成两个随机分布的数据点，分别表示两个类别
class_1 = np.random.rand(data_points, 2) * 2 - 1
class_2 = np.random.rand(data_points, 2) * 2 + 1

# 合并数据点
data = np.vstack((class_1, class_2))

# 生成标签，前50个点属于类别0，后50个点属于类别1
labels = np.hstack((np.zeros(data_points), np.ones(data_points)))

# 定义感知机的权重和偏置
w = np.array([1.5, -0.5])  # 你可以根据需要调整权重
b = -1  # 你可以根据需要调整偏置

# 绘制示意图
plt.scatter(class_1[:, 0], class_1[:, 1], marker='o', label='Class 0')
plt.scatter(class_2[:, 0], class_2[:, 1], marker='x', label='Class 1')

# 添加感知机决策边界
x_values = np.linspace(-2, 4, 100)
y_values = -(x_values * w[0] + b) / w[1]
plt.plot(x_values, y_values, label='Perceptron Decision Boundary', color='red')

# 设置图例和标签
plt.legend()
plt.title('Perceptron Classification Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 显示图形
plt.show()
