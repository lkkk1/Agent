from sklearn.neighbors import KNeighborsRegressor
import numpy as np

X = np.array([[2,1],[3,1],[1,4],[2,6]])
y = np.array([0,1,0,1])

model = KNeighborsRegressor(n_neighbors=2, weights='distance')
model.fit(X,y)

X_pred = np.array([[4,9]])
y_pred = model.predict(X_pred)

# 不设置weights 时， 取最近两个结果的标签均值
# 设置weights 后， 会对两个结果中，距离更近的那个标签进行加权，最终预测结果更靠近距离更近的那个样本的标签
print(y_pred)