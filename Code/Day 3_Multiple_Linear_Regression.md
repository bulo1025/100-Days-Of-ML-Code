# 多元线性回归


<p align="center">
  <img src="https://github.com/MachineLearning100/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%203.png">
</p>


## 第1步: 数据预处理

### 导入库
```python
import pandas as pd
import numpy as np
```
### 导入数据集
```python
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1]
Y = dataset.iloc[ : ,  4 ]
```

### 将类别数据数字化
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X.iloc[: , 3] = labelencoder.fit_transform(X.iloc[ : , 3])
```

### 躲避虚拟变量陷阱
```python
X = X.iloc[: , 1:]
```

### 拆分数据集为训练集和测试集
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
```
## 第2步： 在训练集上训练多元线性回归模型
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```

## Step 3: 在测试集上预测结果
```python
y_pred = regressor.predict(X_test)
```


