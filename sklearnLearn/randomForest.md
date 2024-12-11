# 随机森林
## 1.1 重要参数
### 1.1.1 决策树参数
和决策数一样的参数
![alt text](image-11.png)

### 1.1.2 随机森林参数
1. n_estimators: 树的数量，通常树值越大，精度越高，但是速度也越慢。到一定精度后就不会再提升。

### 1.1.3 举例
实例化
训练集带入实例化后的模型去进行训练：使用的接口是fit 
使用其他接口将测试集导入我们训练好的模型，去获取我们希望获取的结果（score，Y_test）

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
clf.fit(X_train, Y_train)
rfc.fit(X_train, Y_train)

score_c = clf.score(X_test, Y_test)
score_r = rfc.score(X_test, Y_test)

print(score_c)
print(score_r)

# 交叉验证
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc = RandomForestClassifier(nestimators=25)
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)

clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)

plt.plot(range(1,11), rfc_s, label='RandomForest')
plt.plot(range(1,11), clf_s, label='DecisionTree')
plt.legend()
plt.show()
```

# 跟上补充
红色为单个数，蓝色为随机森林，当单个数的准确率低于50%时，随机森林会更差，随机森林的每个分类树至少要有50%的准确率，这样随机森林才能用。
![](image-12.png)