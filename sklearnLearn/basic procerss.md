# 基本流程
![alt text](image.png)
![alt text](image-1.png)
# 交叉验证
将数据划分成n份，每次用n-1份训练，1份做验证，重复n次，取平均值。
![alt text](image-4.png)
cross_val_score()
# np的知识
## 随机数
np.random.RandomState和seed都可以设置随机数，但seed是固定随机数，RandomState是随机随机数。个人理解差别不大，就是seed设置了，后面所有的都得用seed的，RandomState每一截都可以设置不同的种子。
![alt text](image-5.png)
