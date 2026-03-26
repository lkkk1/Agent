import torch
from torch import nn
from torchsummary import summary

X = torch.randn(200,3)

model = nn.Sequential(
	nn.Linear(3, 4),
	nn.Tanh(),
	nn.Linear(4, 4),
	nn.ReLU(),
	nn.Linear(4,2)
)

# 参数初始化
def init_params(layer):
	if isinstance(layer, nn.Linear):
		nn.init.kaiming_uniform_(layer.weight)
		nn.init.constant_(layer.bias, 0.1)

model.apply(init_params)

model.forward(X)
summary(model, input_size=(3,), batch_size=200)