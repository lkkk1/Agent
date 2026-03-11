import numpy as np
from sklearn.feature_selection import VarianceThreshold

# 特征工程 -- 低方差过滤 -- 仅适用于数值特征，对于字符串布尔等非数值特征，不适用！
# 基于特征的方差来筛选，低方差的特征意味着该特征的所有样本值几乎相同，对预测的影响极小，可以将其去掉

# 生成方差接近1的100个随机数
line_a = np.random.randn(100)
print(np.var(line_a))
print(line_a.shape)

# np.random.normal(loc, scale, size)是NumPy中生成正态分布随机数的标准方法
# loc=5: 分布的均值
# scale=0.1: 分布的标准差
# size=100: 生成100个样本
line_b = np.random.normal(5,0.1,size=100)
print(np.var(line_b))
print(line_b.shape)

X = np.vstack((line_a,line_b)).T
print(X.shape)

# 过滤方差低于设定值的特征列
variance_filter = VarianceThreshold(0.01)
X_filtered = variance_filter.fit_transform(X)
print(X_filtered.shape)
print(X_filtered)