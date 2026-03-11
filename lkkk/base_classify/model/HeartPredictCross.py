import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, LabelEncoder
import joblib
from sklearn.model_selection import GridSearchCV

# 1. 读取数据
dataFrame = pd.read_csv('../resource/heart_2020_cleaned.csv', header=0, sep=',')
# print(dataFrame.shape)
X = dataFrame.iloc[:,1:18]
# print(X.shape)
# 这样返回的是第一列，是Series类型
y = dataFrame.iloc[:,0]

# 这样返回的就是DataFrame类型的某一列了， 由于模型期望标签是一维类型，所以不考虑这种
# y = dataFrame.iloc[:,[0]]

# print(y.shape)

# 2. 训练集和测试集的划分
# 应该先进行数据集的划分，后进行特征工程的转换，避免测试数据泄露到训练过程中
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


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

X_trans = column_transformer.fit_transform(X_train)
X_test_trans = column_transformer.transform(X_test)
# KNN模型，要求标签和特征都是数值才行
# map 是Series中对每个元素遍历操作的方法， values将series转为numpy数组，才能作为模型的标签
# y_trans = y.map({'Yes': 1, 'No': 0}).values
Label_encoder = LabelEncoder()
y_trans = Label_encoder.fit_transform(y_train)
y_test_trans = Label_encoder.transform(y_test)
# print(type(y_trans))


# 4. 定义模型
models = {
    "knn": KNeighborsClassifier(),
    "logistic": LogisticRegression(),
}

# 5. 网格搜索
# 对预设的超参数进行组合，找到能使模型最优的参数配置
# 通常内层嵌套交叉验证
results = {}
for name, model in models.items():
    if name == 'knn':
        param_grid = {
            'weights': ["uniform","distance"],
            'n_neighbors': list(range(1,10))
        }
    elif name == 'logistic':
        param_grid = {
            'penalty': ['l2', 'None'],
            'solver': ['lbfgs', 'newton-cg', 'sag', 'saga']
        }
    else:
        param_grid = {}

    # cv是交叉验证的K折数
    # 10000条以内 设置 5 或 10
    # > 10000 , 设置3，减少训练迭代
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    results[name] = {
        'best_params': grid_search.best_params_,    # 最佳训练参数
        'best_cv_score': grid_search.best_score_,  # 这是最佳交叉验证分数
        'best_model': grid_search.best_estimator_,  # 这是最佳参数训练好的模型
    }

    print(f"=== 模型：{name} 网格搜索结果 ===")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")


# 排序选出最好的训练效果对应的模型和参数
# 字典的items() 迭代变量是一个元组（name, result）-- x[1] 就相当于取到对应的result, x[0] 就是name
sorted_results = sorted(results.items(), key=lambda x: x[1]['best_cv_score'], reverse=True)
print(type(sorted_results))

# enumerate 通常结合for, 用于添加排序，1表示从1开始
for index, (name,result) in enumerate(sorted_results, 1):
    print(f"{index}.{name}: {result['best_cv_score']:.4f}")

best_model_name, best_model_result = sorted_results[0]
print(type(sorted_results[0]))

best_model = best_model_result['best_model']


# 6. 在测试集上评估最佳模型
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {test_accuracy:.4f}")

# 7. 模型存储
# 只保存最佳模型（最常用）
joblib.dump(best_model, 'best_knn_model')
print("✓ 已保存最佳模型")

