# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

# 数据加载
filepath = './data/air_data.xls'
data = pd.read_excel(filepath)
# print(data.head())
# print(data.info())      # 无缺失数据

# 数据预处理
index0 = data['空气等级'].unique()
print(index0)
data['空气等级'].replace(data['空气等级'].unique(), [1, 2, 3, 4, 5, 6, 7],inplace=True)
# print(data.head())
print(data['空气等级'].value_counts())
data['空气等级'].replace([7], [6],inplace=True)     # 6 7等级样本合并

features = data.iloc[:, :-1]
labels = data['空气等级']
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=.2, random_state=5)
print(test_y.value_counts())

# 模型建立
model = DecisionTreeClassifier(criterion='gini')
model.fit(train_x, train_y)
pre_y = model.predict(test_x)
cm = confusion_matrix(test_y, pre_y)
print('模型得分:', accuracy_score(test_y, pre_y))

# 混淆矩阵可视化
index0 = index0[:-1]    # 因为6 7级空气等级已经合并, 删除第7级
cm_data = pd.DataFrame(cm, index=index0, columns=index0)
sns.heatmap(cm_data, annot=True, cmap='YlGnBu')
ax = plt.gca()
ax.xaxis.set_ticks_position('top')  # 指定顶端为x轴
plt.xlabel('预测值')
plt.ylabel('实际值')
plt.show()

# 储存决策树模型
dot_data = export_graphviz(model, out_file='air.dot')


