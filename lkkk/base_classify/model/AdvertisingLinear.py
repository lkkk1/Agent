import pandas as pd
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

# 1. 数据读取
df = pd.read_csv('../resource/advertising.csv', header=0, sep=',')
df.dropna(axis=1, inplace=True)

X = df.drop(columns='Sales')
y = df.iloc[:, -1]
# y = df['Sales']
# print(X.shape)
# print(y.shape)

# 2. 特征预处理， 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape)
# print(y_train.shape)
print(y_test.shape)

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

model.fit(X_train, y_train)

linear_model = LinearRegression(
		fit_intercept=True,     # 计算截距，默认true
		tol=1e-6              # 当损失函数改善小于 1e-6的时候，停止迭代
)
linear_model.fit(X_train, y_train)

ridge_model = Ridge(
		alpha=1.0,              # 正则化系数
        fit_intercept=True,     # 计算截距，默认true
        max_iter=1000,
        tol=1e-4,               # 当损失函数改善小于 1e-4的时候，停止迭代
		random_state=42
)
ridge_model.fit(X_train, y_train)


# 模型评估
score = model.score(X_test, y_test)
print(f'model: SGD, score: {score}')
y_pred = model.predict(X_test.iloc[10:11])
print(f'model: SGD, predict: {y_pred}, true: {y_test.iloc[10]}')

linear_score = linear_model.score(X_test, y_test)
print(f'model: LinearRegression, score: {linear_score}')
y_pred_2 = linear_model.predict(X_test.iloc[10:11])
print(f'model: LinearRegression, predict: {y_pred_2}, true: {y_test.iloc[10]}')

ridge_score = ridge_model.score(X_test, y_test)
print(f'model: Ridge, score: {ridge_score}')
y_pred_3 = ridge_model.predict(X_test.iloc[10:11])
print(f'model: Ridge, predict: {y_pred_3}, true: {y_test.iloc[10]}')




