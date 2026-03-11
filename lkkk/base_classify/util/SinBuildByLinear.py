import numpy as np
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.metrics import mean_squared_error  # 均方差损失函数 mse
from sklearn.model_selection import train_test_split  # 划分训练集和验证集
import matplotlib.pyplot as plt
# 将原始特征（比如只有一个 x）转换为多项式特征（比如 [1, x, x^2, x^3, ...]），从而允许线性模型拟合非线性关系。
from sklearn.preprocessing import PolynomialFeatures

# 绘图字体设置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 绘图坐标轴上的符号，不用unicode
plt.rcParams['axes.unicode_minus'] = False

# 1. 创建数据

# 使用NumPy创建了一个包含300个点的数组，数值范围从-3到3，并重塑为一个列向量（形状为 (300, 1)）
X = np.linspace(-3,3, 300).reshape(-1,1)

# np.random.uniform(-0.5, 0.5, 300)创建了一个包含300个随机数的数组，这些随机数在区间 [-0.5, 0.5) 内均匀分布。
# .reshape(-1,1)将这个一维数组重塑为一个列向量，形状为 (300, 1)。
Y = np.sin(X) + np.random.uniform(-0.5,0.5, 300).reshape(-1,1)

fig,ax = plt.subplots(1,3, figsize=(15,4))
ax[0].scatter(X, Y, color='yellow')
ax[1].scatter(X, Y, color='yellow')
ax[2].scatter(X, Y, color='yellow')

# 2. 划分训练集和验证集
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=42)


def below_train(matrix_train_x, matrix_train_y, matrix_test_x, matrix_test_y, model) -> float:
	# 模型训练
	model.fit(matrix_train_x, matrix_train_y)

	# 模型预测，计算误差
	y_pre = model.predict(matrix_test_x)
	print(model.coef_)
	print(model.intercept_)

	linear_train_loss = mean_squared_error(matrix_test_y, y_pre)
	print(linear_train_loss)
	return linear_train_loss

# y=ax+b 欠拟合
x_train1 = train_X
x_test1 = test_X

# 线性回归模型 y=ax+b
linear_model = LinearRegression()
loss = below_train(x_train1, train_Y, x_test1, test_Y, linear_model)
ax[0].plot(X, linear_model.predict(X), 'r')
ax[0].text(-3, 1, f'测试误差：{loss:.4f}', color='red')



def right_train(matrix_train_x, matrix_train_y, matrix_test_x, matrix_test_y, model) -> float:
	model.fit(matrix_train_x, matrix_train_y)
	y_pre = model.predict(matrix_test_x)

	right_train_loss = mean_squared_error(matrix_test_y, y_pre)
	print(right_train_loss)
	return right_train_loss

# 拟合

# 如果 LinearRegression设置了 fit_intercept=True（默认），并且 PolynomialFeatures设置了 include_bias=True，
# 则会产生冗余的截距项。通常建议保持其中一个为True即可。
poly = PolynomialFeatures(degree=3)
x_train2 = poly.fit_transform(train_X)
x_test2 = poly.fit_transform(test_X)

loss2 = right_train(x_train2, train_Y, x_test2, test_Y, linear_model)
ax[1].plot(X, linear_model.predict(poly.transform(X)), 'r')
ax[1].text(-3,1,f'测试误差{loss2:.4f}', color='red')


# 过拟合
poly2 = PolynomialFeatures(degree=20)
x_train3 = poly2.fit_transform(train_X)
x_test3 = poly2.fit_transform(test_X)

def over_train(matrix_train_x, matrix_train_y, matrix_test_x, matrix_test_y, model) -> float:
	model.fit(matrix_train_x, matrix_train_y)
	y_pre = model.predict(matrix_test_x)
	over_train_loss = mean_squared_error(matrix_test_y, y_pre)
	print(over_train_loss)
	return over_train_loss

loss3 = over_train(x_train3, train_Y, x_test3, test_Y, linear_model)
ax[2].plot(X, linear_model.predict(poly2.transform(X)), 'r')
ax[2].text(-3,1, f'测试误差{loss3:.4f}', color='red')

plt.show()