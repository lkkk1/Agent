import torch
import torch.nn as nn
from torchsummary import summary

class TorchNet(nn.Module):
	# 初始化，定义中间层和参数初始化方式
	def __init__(self, device='cpu'):
		super(TorchNet, self).__init__()
		self.linear1 = nn.Linear(3, 4, device=device)
		nn.init.xavier_uniform_(self.linear1.weight)
		self.linear2 = nn.Linear(4, 4, device=device)
		nn.init.kaiming_uniform_(self.linear2.weight)
		self.linear3 = nn.Linear(4, 2, device=device)

	# 前向传播 ， 反向传播由于前向传播记录了值和变化，会自动进行
	def forward(self, x):
		x = self.linear1(x)
		x = torch.tanh(x)

		x = self.linear2(x)
		x = torch.relu(x)

		# softmax 一般用于多分类输出，对于多分类输出，一般是列数对应分类数，dim一般设置为1（列方向压缩）
		x = self.linear3(x)
		x = torch.softmax(x, dim=1)
		return x

# test

# device全局变量
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.randn(100, 3)
model = TorchNet(device)
model.forward(X)

# print(model.linear1.weight.device)

# 查看参数

# torchsummary 查看整体模型架构和参数个数
# imputSize -- 输入维度
# batchSize -- 输入个数
summary(model, input_size=(3,), batch_size=100, device='cpu')

for param in model.parameters():
	print(param)

for name,param in model.named_parameters():
	print(f"name: {name}, param: {param}")

# 字典形式查看模型参数
print(model.state_dict())