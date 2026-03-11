from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"数据集形状: {X.shape}")
print(f"特征: {feature_names}")
print(f"类别: {target_names}")

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 特征标准化（KNN对特征尺度敏感！）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 创建和训练KNN模型
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # 距离加权
    metric='euclidean',
    n_jobs=-1
)
knn.fit(X_train_scaled, y_train)

# 5. 预测和评估
y_pred = knn.predict(X_test_scaled)
y_pred_proba = knn.predict_proba(X_test_scaled)

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("准确率:", knn.score(X_test_scaled, y_test))

# 6. 可视化混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

# 7. 查看最近的邻居
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # 新样本
new_sample_scaled = scaler.transform(new_sample)

distances, indices = knn.kneighbors(new_sample_scaled)
print(f"\n新样本: {new_sample[0]}")
print(f"最近的 {knn.n_neighbors} 个邻居索引: {indices[0]}")
print(f"距离: {distances[0]}")
print(f"邻居的类别: {y_train[indices[0]]}")