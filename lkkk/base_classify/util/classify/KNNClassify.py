import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# KNeighborsClassifier KNN分类模型，取k值个最邻近的点，统计结果作为预测结果
# 设置weights = ‘distance’ 设置权重为距离，当最邻近的k个样本的标签不同时，距离更近的权重更大

# 1. 生成数据
X = np.array([[3,1],[4,1],[2,9],[3,8]])

# 4 * 2
print(X.shape)
y = np.array([1,0,1,0])

# 2. 定义模型，训练数据
# weights 设置权重，邻近的2个点分类不同时，通过权重区分选择，距离最近的点的结果作为最终结果
model = KNeighborsClassifier(n_neighbors=2, weights='distance')
model.fit(X,y)

# 3. 预测
X_pred = np.array([[5,13]])
y_pred = model.predict(X_pred)
print(y_pred)

# 4. 绘图描述
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
# 设置坐标轴相等
ax.axis('equal')

# 借助numpy的布尔索引功能，筛选数据
X_1 = X[y == 0]
X_2 = X[y == 1]

# 'C0' 这些，有默认提供的颜色可以使用
colors = ['C0', 'C1']

# 4 * 2 , 取列值 对应坐标点，行表示的是个数，取：
ax.scatter(X_1[:,0],X_1[:,1], c=colors[0])
ax.scatter(X_2[:,0],X_2[:,1], c=colors[1])

color_pred = colors[0] if y_pred == 0 else colors[1]
ax.scatter(X_pred[:,0], X_pred[:,1], c=color_pred)

plt.show()
plt.close(fig)



