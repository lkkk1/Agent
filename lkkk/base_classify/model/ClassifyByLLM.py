import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 1. 数据读取
df = pd.read_csv('../resource/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv', header=0, sep=',')
# 查看前几行
# print(df.head(1))
# # 查看数据形状（行，列）
# print(df.shape)
# # 查看列名
# print(df.columns)
# # 查看索引
# print(df.index)
# # 查看基本信息
# print(df.info())
# # 快速统计描述
# print(df.describe())

# 选择数据
keys = df.loc[:,'text']
# print(keys.head(1))
# print(type(keys))

# dataframe 还支持分组聚合（groupby）, 排序（sort_values）,应用函数（apply）等

# 2. 数据预处理 -- 数据分词，清洗，去停用词
input_keys = keys.apply(lambda x: " ".join(jieba.lcut(x)))

# 最终仍需要转为字符串，而不能是列表
# input_keys = keys.apply(lambda x: jieba.lcut(x))
# print(input_keys.head(5))

# print(type(input_keys))
# print(input_keys.head(3))
# print(input_keys.shape)

# 分词后，仍然是series类型，有索引，要提取转为普通的列表（仅保留分词结果，去掉索引），通过values转为ndarray
# arrays = input_keys.loc[0:1].values
# print(type(arrays))
input_arrays= input_keys.values
# print(input_arrays.shape)
# print(type(input_arrays[0]))

# 3. 特征提取
# vectorizer = CountVectorizer()

# 不当的去除罕见词设置确实可能导致文本失去关键特征。这是一个需要谨慎权衡的问题。
# 去除罕见词的好处：减少噪声、降低维度、提高效率
# 风险：可能丢失重要的低频关键词（如专业术语、人名、特定实体）

vectorizer = CountVectorizer(
    max_df=0.8,        # 忽略在80%以上文档中出现的词（去除常见词）
	min_df=2,          # 忽略少于1个文档中出现的词（去除罕见词）
    max_features=5000, # 只保留前1000个最常见的词
    vocabulary=None    # 可手动指定词汇表
)

# 学习并转换为特征矩阵（基于词频，根据分词结果统计的词频稀疏矩阵）
# CountVectorizer接受的输入是一组文本文档，上边通过jieba分词后，再用空格串联起文本，最终仍组合为一个有空格分词的字符串输入，
# jieba只是辅助了分词过程，最终由vectorizer完成向量转换和词频统计学习
input_matrix = vectorizer.fit_transform(input_arrays)
# print(input_matrix.shape)  # 3719 ， 去罕见词 1786
# 处理后的词汇表
# print(vectorizer.get_feature_names_out())
# 词频统计结果
# print(vectorizer.vocabulary_)
# print(type(input_matrix))

# 3. 模型训练
# label 类型有8种， 设置neighbors -- 8+3,奇数有时候可以避免平票
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(input_matrix, df.loc[:,'label'])

# 4. 模型预测
str_to_predict = "我要好好爱自己！做自己的一束光！"
predict_input = vectorizer.transform([" ".join(jieba.lcut(str_to_predict))])
res = classifier.predict(predict_input)
print(res)


# Q
# 1. KNN的原理，影响因素，数据多少会对准确性有影响吗
# 2. 数据分割，训练集和测试集
# 3. KNN其他的参数设置，n-gram以及距离单位？
# 4. 为什么训练不需要对label转id，是在模型内部转换了吗
# 5. llm训练和预测
# 6， 为什么KNN不需要对输入做标准化处理，比如截取相同长度？
# 7. 训练后，模型会有记忆吗？如果同一批训练集不断重新训练，会不断优化吗？这个训练的结果需要存储起来吗？
# 8. matlab绘图表示
