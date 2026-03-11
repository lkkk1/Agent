import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler

X = np.array([[2,1],[3,1],[1,4],[2,6],[1000,999]])
print(X.shape)

# 归一化 -- 会受到噪声值干扰。计算转换得到一个范围结果
# 适合有边界的情况，如图像像素，词频等
# feature_range 设置转换后的归一化范围
scaler = MinMaxScaler(feature_range=(-1,1))
X_trans = scaler.fit_transform(X)
print(X_trans)

# 标准化 -- 少量噪声的影响更小，鲁棒性更好。 -- 转换为均值为0，标准差为1的正态分布结果
# 适合范围不定，或有少量噪声的通用处理
standard_scaler = StandardScaler()
X_trans2 = standard_scaler.fit_transform(X)
print(X_trans2)

# 鲁棒标准化，计算时，取25% - 75% 范围的统计量，对异常值有更好的鲁棒性
robust_scaler = RobustScaler(quantile_range=(25, 75))
X_trans3 = robust_scaler.fit_transform(X)
print(X_trans3)

