import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

def create_dataset():
	data_frame = pd.read_csv("../resource/fashion-mnist_test.csv", header=0, sep=",")
	X = data_frame.iloc[:, 1:]
	y = data_frame.iloc[:, 0]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# N * C * m * n -- C 图层通道数
	X_train = torch.tensor(X_train.values, dtype=torch.float).reshape(-1, 1, 28, 28)
	X_test = torch.tensor(X_test.values, dtype=torch.float).reshape(-1, 1, 28, 28)

	y_train = torch.tensor(y_train.values, dtype=torch.int64)
	y_test = torch.tensor(y_test.values, dtype=torch.int64)

	train_dataset = TensorDataset(X_train, y_train)
	test_dataset = TensorDataset(X_test, y_test)

	return train_dataset, test_dataset

# 1. 数据读取，构建数据集
train_dataset, test_dataset = create_dataset()

# X, y = train_dataset[123]
# X = X.permute(1,2,0)
# print(type(X))
# print(X.shape)
# print(type(y))
# print(y.numpy())

# pyTorch 的图像张量通常是 [batch_size, channels, height, width]格式：
# 你的形状：[1, 1, 28, 28]
# 含义：batch_size=1, channels=1, height=28, width=28
# 但 imshow需要：[height, width]或 [height, width, channels]

# plt.imshow(X, cmap='gray')
# plt.show()
# print(y)

# 2. 模型定义
model = nn.Sequential(
	# N,1,28,28
	nn.Conv2d(1, 6, 5, 1,2),
	nn.Sigmoid(),
	# N,6,28,28
	nn.AvgPool2d(2, 2, 0),

	# N,6,14,14
	nn.Conv2d(6, 16, 5, 1,0),
	nn.Sigmoid(),
	# N,16,10,10
	nn.AvgPool2d(2, 2, 0),

	# N，16，5，,5
	## 展平层，多维向量展平为一维
	nn.Flatten(),
	# N，400
	nn.Linear(400, 120),
	nn.Sigmoid(),
	# N,120
	nn.Linear(120, 84),
	nn.Sigmoid(),
	# N,84
	nn.Linear(84, 10)
	# N,10
)

# 3. 模型训练，测试
def train_test(model, train_dataset, test_dataset, epoch_num, lr, batch_size, device):
	# 参数初始化函数
	def init_weights(layer):
		if (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
			# 因为激活函数是sigmoid ， 使用泽维尔均匀初始化，每一层根据输入和输出的维度调整参数，缓解梯度消失和梯度爆炸问题
			torch.nn.init.xavier_uniform_(layer.weight)
	model.apply(init_weights)
	model.to(device)

	# 定义损失函数, 优化器
	loss = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	for epoch in range(epoch_num):
		model.train()
		train_loss = 0
		train_correct_num = 0

		# 定义dataLoader
		data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		for batch_idx, (data, target) in enumerate(data_loader):
			data, target = data.to(device), target.to(device)
			# 前向传播
			predict = model(data)
			# 计算损失
			loss_val = loss(predict, target)
			# 反向传播
			loss_val.backward()
			# 梯度更新
			optimizer.step()
			# 梯度清零
			optimizer.zero_grad()

			train_loss += loss_val.item() * data.shape[0]
			# 统计正确率， 取模型输出最大值的索引，对应分类
			output = predict.argmax(dim=1)
			train_correct_num += output.eq(target).sum()

			# 打印进度条
			print(f'\rEpoch:{epoch+1:0>2}[{"="* int(batch_idx/len(data_loader)*50)}]', end='')

		# 计算本轮平均的损失和正确率
		this_loss = train_loss / len(train_dataset)
		this_train_acc = train_correct_num / len(train_dataset)


		# 测试验证
		model.eval()
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
		test_correct_num = 0
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)

				# 前向传播，预测
				test_predict = model(data)
				output = test_predict.argmax(dim=1)
				test_correct_num += output.eq(target).sum()
		this_test_acc = test_correct_num / len(test_dataset)

		print(f'train_loss:{this_loss:.4f}, train_acc:{this_train_acc:.4f}, test_acc:{this_test_acc:.4f}')


# 超参数定义，调用训练测试方法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.01
batch_size = 256
epoch_num = 20

train_test(model, train_dataset, test_dataset, epoch_num, lr, batch_size, device)

# 4. 选一个数据做测试对比
x_demo,y_demo = test_dataset[123]
print(x_demo.shape)
x_demo_trans = x_demo.permute(1,2,0)

print(y_demo.item())
pred = model(x_demo.unsqueeze(dim=0).to(device))
output = pred.argmax(dim=1).item()
print(f'output:{output}')

plt.imshow(x_demo_trans, cmap='gray')
plt.show()
