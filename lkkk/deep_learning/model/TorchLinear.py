import torch
from torch import nn, optim  # 模型和优化器
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# torch 机器学习拟合线性模型

# 1. 构建数据
# 均值为0，方差为1，（100，1）的张量
X = torch.randn(100, 1)
# 预设系数
w = torch.tensor([2.5])
b = torch.tensor([5.2])
# 定义随机噪声
noise = torch.randn(100, 1) * 0.5

# 要拟合的数据y
y = w * X + b + noise

# TensorDataset, DataLoader
data_set = TensorDataset(X, y)

# 对总的dataset数据集，划分batch_size个小批量数据集，每个数据，都是（x,y）的元组
data_loader = DataLoader(data_set, batch_size=100, shuffle=True)

# 2. 构建模型
# 模型需要指定，输入输出的数据维度
model = nn.Linear(1,1)

# 3. 定义损失函数和优化器
# 损失函数
loss_fn = nn.MSELoss()
# 梯度迭代优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练数据
epoch_num = 300
loss_list = []

# 每个轮次
for epoch in range(epoch_num):
	total_loss = 0  # 本轮总损失
	iter_num = 0   # 本轮迭代次数
	# 每个轮次内，每轮迭代，以batch_size为边界，做一次梯度迭代
	for x_train, y_train in data_loader:
		# 前向传播
		y_pred = model.forward(x_train)
		# 计算损失
		loss_val = loss_fn(y_pred, y_train)
		# 考虑，如果输入数据总数，不能整除batch_size, 最后一个批次数量可能不足batch_size , 所以在total_loss统计的时候，乘以实际个数
		total_loss += loss_val.item() * x_train.shape[0]
		iter_num += 1
		# 反向传播
		loss_val.backward()
		# 更新梯度
		optimizer.step()
		# 梯度清零
		optimizer.zero_grad()
	# 计算本轮平均损失
	loss_list.append(total_loss / len(data_loader))

print(model.weight)
print(model.bias)

# 画图
fig, ax = plt.subplots(1,2,figsize=(14,5))
# 1. 训练损失，随epoch的变化
ax[0].plot(loss_list)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
# 2. 绘制散点图，和拟合直线
ax[1].scatter(X,y)
y_pre = model.weight.item() * X + model.bias.item()
ax[1].plot(X, y_pre, color="red")
plt.show()












