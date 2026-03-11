import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, LabelEncoder
import joblib

# 1. 读取数据
dataFrame = pd.read_csv('/Users/lkkk/code/github/Agent/lkkk/base_classify/resource/heart_2020_cleaned.csv',header=0, sep=',')
# print(dataFrame.shape)
X = dataFrame.iloc[:,1:18]
# print(X.shape)
# 这样返回的是第一列，是Series类型
y = dataFrame.iloc[:,0]

# 这样返回的就是DataFrame类型的某一列了， 由于模型期望标签是一维类型，所以不考虑这种
# y = dataFrame.iloc[:,[0]]

# print(y.shape)

# 2. 特征工程
# 二分类特征 -- Yes/No 转 1/0
# 多分类特征 -- 独热编码 + drop first -- 避免多重共线性
# 连续值 -- 标准化

# 二分类 转换0/1
binary_feature_sex = ['Sex']
binary_features_yn = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma',
    'KidneyDisease', 'SkinCancer']

# 多分类 -- 独热编码+drop first
multi_classify_features = ['AgeCategory', 'Race', 'GenHealth', 'Diabetic']

# 连续数值型 -- 标准化
continuous_features = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']

def trans_yn_feature(X_yn_feature):
    X_yn = X_yn_feature.copy()
    return X_yn.replace({'Yes': 1, 'No': 0})

# for feature in multi_classify_features:
#     unique_val = X[feature].unique()
#     print(f'{feature} 唯一值: {unique_val}')
#
# for feature in continuous_features:
#     unique_val = X[feature].unique()
#     print(f'{feature} 唯一值: {unique_val}')

# print("=== Diabetic列的值分布 ===")
# unique_val = X['Diabetic'].unique()
# print(unique_val)


# FunctionTransformer将自定义函数转换成 scikit-learn 的转换器接口，使其可以：
# 无缝集成到 Pipeline 中
# 在 ColumnTransformer 中使用
# 支持 fit/transform 接口
# 支持交叉验证和网格搜索
column_transformer = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(drop='first'), multi_classify_features),
    ('standard', StandardScaler(), continuous_features),
    ('binary_sex', FunctionTransformer(lambda X: X.replace({'Male': 1, 'Female': 0})), binary_feature_sex),
    ('binary_yn', FunctionTransformer(trans_yn_feature), binary_features_yn),
])

X_trans = column_transformer.fit_transform(X)
# KNN模型，要求标签和特征都是数值才行
# y_trans = y.map({'Yes': 1, 'No': 0}).values
# print(type(y_trans))

# OrdinalEncoder -- 有序特征，维持次序在编码中体现: 专门为多列特征设计, sklearn 可用于特征编码
# OneHotEncoder -- 独热编码，将多分类特征转换为二进制特征矩阵，每个类别对应一个二进制列， sklearn 可用于特征编码

# LabelEncoder sklearn 专门对标签列编码的
Label_encoder = Label
Encoder()
y_trans = Label_encoder.fit_transform(y)
print(type(y_trans))

# 3. 训练集和测试集的划分
X_train, X_test, y_train, y_test = train_test_split(X_trans,y_trans,test_size=0.2,random_state=42)

# 4. 定义模型
model = KNeighborsClassifier(n_neighbors=3, weights='distance')

# 5. 模型训练
model.fit(X_train,y_train)

# 6. 模型评估
score = model.score(X_test,y_test)
print(score)

# 7. 模型存储
joblib.dump(model, 'knn_heart_predict_model')

# 8. 加载模型，对新数据进行模型预测
knn_load = joblib.load('knn_heart_predict_model')
y_pred = knn_load.predict(X_test[10:11])
print(f'预测类别：{y_pred}, 真实类别：{y_test[10]}')
