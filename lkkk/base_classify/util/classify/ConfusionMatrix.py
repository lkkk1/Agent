import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from matplotlib import pyplot as plt

label = ['猫', '狗']
y_pred = ['猫','狗','狗','猫','狗','猫','狗','狗','狗']
y_true = ['猫','狗','狗','猫','狗','狗','狗','猫','猫']

confusion_matrix = confusion_matrix(y_true, y_pred)
# print(confusion_matrix)
print(pd.DataFrame(confusion_matrix, index=label, columns=label))

sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='coolwarm')
plt.show()

# 准确率 -- 预测正确的/总数
accuracy = accuracy_score(y_true, y_pred)
print(f'准确率：: {accuracy}')

# 精确率 -- 在预测的正例范围内，预测正确的/预测是正例的总数
precision = precision_score(y_true, y_pred, pos_label='狗')
print(f'精确率: {precision}')

# 召回率 -- 在真实的正例范围内，预测正确的/真实是正例的总数
recall_score = recall_score(y_true, y_pred, pos_label='狗')
print(f'召回率: {recall_score}')

# 精确率和召回率的调和平均 = 2 *（精确率 * 召回率）/ 精确率 + 召回率
f1_score = f1_score(y_true, y_pred, pos_label='狗')
print(f'f1_score: {f1_score}')

# labels 设置所有标签类型，
# target_names 设置关注的正类类别，可以设置多个，适用于多分类任务，如果设置为None，表示所有的类别都是关注的正类类别，都会分别计算输出
classification_report = classification_report(y_true, y_pred, labels=label, target_names=None)
print(classification_report)