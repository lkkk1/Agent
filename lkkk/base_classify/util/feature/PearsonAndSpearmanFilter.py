import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Pearson / Spearman 相关系数过滤 ， 仅适用于 特征和标签都是数值的数据，提取特征和标签之间的关联关系

data_frame = pd.read_csv('../../resource/data.csv', header=0, sep=',')
print(data_frame.shape)
# print(data_frame.head(0))
# print(data_frame.describe())

X_matrix = data_frame.loc[:,['题名','责任者','ISBN','借出','还回','条码','馆藏点']]
print(X_matrix.shape)
Y_matrix = data_frame.loc[:,'索书号']
print(Y_matrix.shape)

# 通过dataframe的corrwith，获取特征和标签之间的关联关系，关联越紧密，数值越大
# 也可以通过dataframe的corr，从整个完整数据集data_frame中，各个列之间两两获取关联关系。

# pearson相关系数（数值大小之间关联）
# spearman相关系数（数值单调性之间关联）
# 二者均用于数值类型的特征和标签，之间统计关联关系
relate_matrix = X_matrix.corrwith(Y_matrix, method='pearson')
print(relate_matrix)

# 绘制热力图
sns.heatmap(relate_matrix, annot=True, cmap="coolwarm", fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()



