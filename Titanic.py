import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier


# STEP 1. 读取并探索数据

# 读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.info())  # 了解数据表的基本情况
print('-'*30)

print(train_data.describe())  # 了解数据表的统计情况
print('-'*30)

print(train_data.describe(include=['O']))  # 计算离散型变量的统计特征
print('-'*30)

print(train_data.head())  # 查看首5行
print('-'*30)

print(train_data.tail())  # 查看末5行


# STEP 2. 清洗数据，处理缺失值

# 使用平均年龄来填充年龄中的 NaN 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

# 使用票价的均值填充票价中的 NaN 值
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# 对 train_data 的 Embarked 的不同取值计数，找出港口的众数
print(train_data['Embarked'].value_counts())

# 使用港口的众数(S)来填充登录港口的 NaN 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)


# STEP 3. 选择特征，特征向量化

# 选择特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# 抽取特征
x_train = train_data[features]
x_test = test_data[features]
y_train = train_data['Survived']

# 特征向量化：将分类数据转为数值数据(原数值数据不变)

# 初始化 DictVectorizer 特征抽取器(不产生稀疏矩阵)
dict_vec = DictVectorizer(sparse=False)

# 转换(先转为字典，再进行转换)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
x_test = dict_vec.transform(x_test.to_dict(orient='record'))

# 输出各个维度的特征含义
print(dict_vec.feature_names_)


# STEP 4. 建立决策树模型并训练

# 构造 ID3 决策树(entropy:ID3; gini:CART)
clf = DecisionTreeClassifier(criterion='entropy')

# 训练决策树
clf.fit(x_train, y_train)


# STEP 5. 进行预测，写入文件

# 决策树预测
y_pred = clf.predict(x_test)

# 写入 gender_submission.csv
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})
output.to_csv('gender_submission.csv', index=False)
