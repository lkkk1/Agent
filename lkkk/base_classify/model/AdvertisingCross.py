import pandas as pd
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score

# 1. 数据读取
df = pd.read_csv('../resource/advertising.csv', header=0, sep=',')
df.dropna(axis=1, inplace=True)

X = df.drop(columns='Sales', axis=1)
# y = df['Sales']
y = df.iloc[:, -1]
# y = df['Sales']
# print(X.shape)
# print(y.shape)

# 2. 特征预处理， 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape)
# print(y_train.shape)
# print(y_test.shape)

# 3. 模型定义

# SGD -- 梯度下降计算线性回归 -- 适用于大批量数据训练，数据量小时，需要提高迭代次数，降低学习率
model = SGDRegressor(
		loss="squared_error",   # 损失函数 -- 均方差
        penalty="l2",           # l2 正则化
        alpha=0.0001,           # 正则化系数
        fit_intercept=True,     # 计算截距，默认true
        max_iter=1000000,          # 最大迭代次数
        tol=1e-3,               # 当损失函数改善值小于 1e-3的时候，停止迭代
		eta0=0.0001,              # 学习率
)

linear_model = LinearRegression(
		fit_intercept=True,     # 计算截距，默认true
		tol=1e-6              # 当损失函数改善小于 1e-6的时候，停止迭代
)

ridge_model = Ridge(
		alpha=1.0,              # 正则化系数
        fit_intercept=True,     # 计算截距，默认true
        max_iter=1000,
        tol=1e-4,               # 当损失函数改善小于 1e-4的时候，停止迭代
		random_state=42
)

models = {
	"SGD": model,
	"Ridge": ridge_model,
	"LinearRegression": linear_model
}

results = {}

# 回归问题，scoring = r2 决定系数，对于分类问题，一般可以用 accuracy
for name,model in models.items():
	# 交叉验证过程中确实会训练模型，但这些训练仅用于评估目的，并且每次交叉验证折都使用不同的训练子集。
	# 当交叉验证完成后，这些临时训练的模型会被丢弃，原始模型对象仍处于未拟合状态。
	scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
	results[name] = {
		'mean_score': scores.mean(),
		'std_score': scores.std()
	}
	print(f"{name:20}: {scores.mean():.3f} ± {scores.std():.3f}")

# for name,res in results.items():
# 	print(f"{name:20}: {res}")

# 模型预测
for name,model in models.items():
	# 交叉验证只是为了评估模型在训练集上的泛化性能，通常可用于选择模型阶段进行比较，但不会修改原始模型对象，
	# 因此仍未拟合，如果要做测试集训练，仍需要重新拟合
	model.fit(X_train, y_train)
	score = model.score(X_test, y_test)
	print(f"{name:20}: {score:.3f}")
