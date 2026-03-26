import torch
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# 1. 数据读取
def create_dataset():
	data_frame = pd.read_csv('../resource/house_price.csv', header=0, sep=',')
	data_frame.drop(['Id'], axis=1, inplace=True)
	# X = data_frame.drop('SalePrice', axis=1, inplace=False)
	X = data_frame.iloc[: , 0:-1]
	# y = data_frame['SalePrice']
	y = data_frame.loc[: , ['SalePrice']]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# 特征工程
	# 根据特征的数据类型，将特征划分为数值型和类别型
	num_features = X.select_dtypes(exclude=['object']).columns
	classify_features = X.select_dtypes(include=['object']).columns

	# 类别转换器
	num_feature_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='mean')),   # 缺省值填充，使用mean 平均值填充
		('scaler', StandardScaler())                   # 标准化处理
	])

	class_feature_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),  # 缺省值填充，固定值 NaN
		('encoder', OneHotEncoder(handle_unknown='ignore'))  # 类别独热编码，对于测试集中出现训练集中没有的类别，ignore; 测试集仅transform，没有fit
	])

	data_transformer = ColumnTransformer(transformers=[
		('num_features', num_feature_transformer, num_features),
		('classify', class_feature_transformer, classify_features)
	])

	X_train = data_transformer.fit_transform(X_train)
	X_test = data_transformer.transform(X_test)
	print(type(X_train))

	# columnTransformer 转换后，由于分类特征进行独热编码，转换后是一个稀疏矩阵，由于分类特征较多，所以考虑通过toarray() 将其转为稠密矩阵
	# X_train = pd.DataFrame(X_train.toarray(), columns=data_transformer.get_feature_names_out())
	# X_test = pd.DataFrame(X_test.toarray(), columns=data_transformer.get_feature_names_out())
	X_train_dense = X_train.toarray()
	X_test_dense = X_test.toarray()
	print(type(X_train_dense))

	# 构建TensorDataSet, values是将pd 的 dataframe 转为ndarray ， 而经过columnTransformer转换和toarray转换后，已经是ndarray了，不需要转pd.df了, tensor方法根据内容创建
	# train_dataset = TensorDataset(torch.tensor(X_train.values).float(), torch.tensor(y_train.values).float())
	# test_dataset = TensorDataset(torch.tensor(X_test.values).float(), torch.tensor(y_test.values).float())

	# y_train 和 y_test 还是pd.df ，需要values 转为ndarray
	train_dataset = TensorDataset(torch.tensor(X_train_dense).float(), torch.tensor(y_train.values).float())
	test_dataset = TensorDataset(torch.tensor(X_test_dense).float(), torch.tensor(y_test.values).float())

	return train_dataset, test_dataset, X_train_dense.shape[1]

train_dataset, test_dataset, feature_num = create_dataset()
# print(feature_num)

# 2. 模型定义
model = nn.Sequential(
	nn.Linear(feature_num, 128),
	# 批量标准化， 针对每列特征，做标准化处理，可以提高收敛速度，提升泛化能力
	nn.BatchNorm1d(128),
	nn.ReLU(),
	# dropout 训练时关闭20%的节点，同比增大剩余节点的输出，保持输出平衡，提高模型泛化能力
	nn.Dropout(0.2),
	nn.Linear(128, 1)
)

# 3. 自定义损失函数 -- 房价是一个较大的数值，对于回归问题，通常使用mse均方误差，会更放大这个本来就很大的值，因此考虑log + 开方处理
def log_mse_loss(y_pred, y_true):
	# 截取 1到正无穷的部分，这样log才能求得正数
	y_pred = torch.clamp(y_pred, 1, float('inf'))
	loss = nn.MSELoss()
	return torch.sqrt(loss(torch.log(y_pred), torch.log(y_true)))


# 4. 模型训练
def train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
	# 模型参数初始化
	def init_params(layer):
		if isinstance(layer, nn.Linear):
			nn.init.kaiming_normal_(layer.weight)
	model.apply(init_params)

	# 模型加载到设备
	model.to(device)

	# 定义优化器
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	train_loss_list = []
	test_loss_list = []

	# 训练
	for epoch in range(epoch_num):
		# 切换到训练模式
		model.train()
		train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		total_loss = 0
		for batch_idx, (X, y) in enumerate(train_data_loader):
			X, y = X.to(device), y.to(device)
			# 前向传播
			y_pred = model(X)
			# 计算损失，squeeze 降维计算 ， y_pred是一个【batch_size, 1】的二维张量，squeeze 移除张量中所有1维的维度，即转为【batch_size】的一维张量，和y_true维度一致
			loss_val = log_mse_loss(y_pred.squeeze(), y)
			loss_val.backward()
			optimizer.step()
			optimizer.zero_grad()

			total_loss += loss_val.item() * X.shape[0]

		this_train_loss = total_loss / len(train_dataset)
		train_loss_list.append(this_train_loss)

		# 测试
		model.eval()
		test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
		total_loss = 0
		with torch.no_grad():  # 不计算更新梯度
			for X, y in test_data_loader:
				X, y = X.to(device), y.to(device)
				y_pred = model(X)
				loss_val = log_mse_loss(y_pred.squeeze(), y)
				total_loss += loss_val.item() * X.shape[0]
			this_test_loss = total_loss / len(test_dataset)
			test_loss_list.append(this_test_loss)

		print(f'epoch: {epoch+1}, train_loss: {this_train_loss}, test_loss: {this_test_loss}')

	return train_loss_list, test_loss_list

# 超参数定义
lr = 0.1
epoch_num = 200
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_list, test_loss_list = train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device)

# 画图
plt.plot(train_loss_list, label='train loss', color='blue', linewidth=2, linestyle='-')
plt.plot(test_loss_list, label='test loss', color='red', linewidth=3, linestyle='-.')
plt.legend()
plt.show()










