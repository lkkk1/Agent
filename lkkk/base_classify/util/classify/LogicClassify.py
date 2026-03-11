from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# 1. 生成数据
# 对于多分类问题，通常设置 n_clusters_per_class = 1
# 对于2分类问题，通常设置 n_clusters_per_class = 2

# n_informative 表示有信息特征数，可以设置适当多一点(如： feature数量/2 + 1)
# X,y = make_classification(n_samples=5000, n_classes=3, n_features=30, n_informative=16, n_clusters_per_class=1, random_state=42)
X,y = make_classification(n_samples=5000, n_classes=2, n_features=30, n_informative=16, n_clusters_per_class=2, random_state=42)


# print(X.shape)
# print(y.shape)

# 2. 训练集和测试集拆分
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. 设置模型
model = LogisticRegression()

# 4. 训练
model.fit(train_X, train_Y)

# 5. 预测
y_pred = model.predict(test_X)

# 6. 评估训练结果
report = classification_report(test_Y, y_pred)
print(report)

# 7. 获取模型预测概率, 计算auc值（auc适合二分类任务评估模型，值越大，模型性能越好）
# 保留取正类的概率值，二分类，0/1，正类是1，所以取第二列
y_pred_proba = model.predict_proba(test_X)[:, 1]
print(y_pred_proba.shape)

roc_auc = roc_auc_score(test_Y, y_pred_proba)
print(roc_auc)



