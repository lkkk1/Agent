import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

# 1. 加载数据
dataFrame = pd.read_csv('/lkkk/base_classify/resource/heart_2020_cleaned.csv', header=0, sep=',')
# print(dataFrame.shape)
X = dataFrame.iloc[:,1:18]
# print(X.shape)
# 这样返回的是第一列，是Series类型
y = dataFrame.iloc[:,0]

# 2. 特征转换
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

column_trans = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(drop='first'), multi_classify_features),
    ('standard', StandardScaler(), continuous_features),
    ('binary_sex', FunctionTransformer(lambda X: X.replace({'Male': 1, 'Female': 0})), binary_feature_sex),
    ('binary_yn', FunctionTransformer(trans_yn_feature), binary_features_yn),
])

X_trans = column_trans.fit_transform(X)
y_trans = y.map({'Yes': 1, 'No': 0}).values

knn_load = joblib.load('best_heart_predict_model')
y_pred = knn_load.predict(X_trans[5:6])
print(f'预测类别：{y_pred}, 真实类别：{y_trans[5]}')


