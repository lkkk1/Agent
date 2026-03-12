import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. 读取数据
df = pd.read_csv('../resource/train.csv', header=0, sep=',', nrows=5000)
df.dropna(inplace=True, axis=1)

X = df.drop(columns=['label'], axis=1)
y = df['label']
# print(y.shape)
# print(X.shape)

# 测试图像
# digit = X.iloc[0, :].values.reshape(28,28)
# print(type(digit))
# plt.imshow(digit, cmap='Greys_r')
# plt.show()

# 2. 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. 特征处理 -- rgb数值，有范围，可以用归一化
scaler = MinMaxScaler()
X_train_trans = scaler.fit_transform(X_train)
X_test_trans = scaler.transform(X_test)

# 网格搜索， 模型训练
models = {
	# OVR 即将多分类，在底层转为N个二分类问题，每个分类器对应一个类别作为正类，其他类别作为负类计算概率，综合所有分类器的结果
	"ovr": OneVsRestClassifier(LogisticRegression(max_iter=2000)),
	# 逻辑线性回归，默认底层就是softmax函数将线性回归结果通过softmax函数映射到分类概率，其他的映射函数如sigmoid
	"softmax": LogisticRegression(max_iter=2000),
}

results = {}
for name, model in models.items():
	grid = GridSearchCV(estimator=model, cv=3, param_grid={'solver': ['lbfgs', 'saga']})
	if name == "ovr":
		# 对于OneVsOneClassifier，需要指定内部estimator的参数
		grid = GridSearchCV(estimator=model, cv=3, param_grid={'estimator__solver': ['lbfgs', 'saga']})
	else:
		# 对于普通的LogisticRegression，直接使用solver参数
		grid = GridSearchCV(estimator=model, cv=3, param_grid={'solver': ['lbfgs', 'saga']})

	grid.fit(X_train_trans, y_train)
	results[name] = {
		"best_params": grid.best_params_,
		"best_score": grid.best_score_,
		'best_estimator': grid.best_estimator_,
	}
	print(f'model: {name}, Best params: {grid.best_params_}')
	print(f'model: {name}, Best score: {grid.best_score_}')

sorted_results = sorted(results.items(), key=lambda x: x[1]['best_score'], reverse=True)
for index, (name, res) in enumerate(sorted_results, 1):
	print(f'{index}. model: {name}, score: {res["best_score"]:.3f}')

best_model  = sorted_results[0][1]['best_estimator']

# 4. 模型评估
test_score = best_model.score(X_test_trans, y_test)
print(f'Test score: {test_score:.3f}')

# X_test_trans 在经过 MinMaxScaler 变换后已经变成了 numpy 数组
y_pred = best_model.predict(X_test_trans[10:11])
print(f'y_pred: {y_pred}')
print(f'y_test: {y_test.iloc[10]}')

# X_test_trans 在经过 MinMaxScaler 变换后已经变成了 numpy 数组, 不需要再values转了。dataframe才需要values转为ndarray
plt.imshow(X_test_trans[10:11].reshape(28, 28), cmap='Greys_r')
plt.show()

# 5. 模型保存
joblib.dump(best_model, '../resource/best_model_for_digit')

# 用心走过的路，认真爱过的人，全情投入去做的事 才是人生最宝贵的价值，而非其他。