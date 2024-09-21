

# 1. 环境配置  

## 1.1 使用colab.research.google.com真的好用，赞！  
## 1.2 ctrl+m m转换代码为文本  
## 1.3  导包几件套  
import torch  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

# 2. 常用命令  

## 2.1 标量，比如torch.tensor(2)  
### 2.1.1 维度ndim  
scalar = torch.tensor().ndim，标量没有维度  
### 2.1.2 获取原本数值  
torch.tensor().item()  

## 2.2 向量（矢量）
比如torch.tensor([7,7])  
### 2.2.1 ndim  
ndim为1
### 2.2.2 shape
shape为2，为向量里面的元素个数  

## 2.3 矩阵Matrix， 比如torch.tensor
([[7, 8]  
    [9, 10]])  
### 2.3.1 ndim 为2  
### 2.3.2 切片[0], torch.tensor([7, 8])  
### 2.3.3 shape 为 torch.Size([2, 2])  

## 2.4 张量 tensor
例TENSOR = torch.tensor(  
    [[[1, 2, 3],  
    [3, 6, 9],  
    [2, 4, 5]]]  
)
### 2.4.1 TENSOR.ndim = 3
### 2.4.2 TENSOR.shape = torch.Size(1,3,3)
![alt text](image.png)