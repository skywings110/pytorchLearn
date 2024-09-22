

# 1. 环境配置  

## 1.1 使用colab.research.google.com真的好用，赞！  
## 1.2 ctrl+m m转换代码为文本  
## 1.3  导包几件套  
```python
import torch  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
```
# 2. 标量、向量、矩阵、张量  

## 2.1 标量，比如torch.tensor(2)  
### 2.1.1 维度ndim  
```python
scalar = torch.tensor().ndim，标量没有维度  
```
### 2.1.2 获取原本数值  
```python
torch.tensor().item()  
```
## 2.2 向量（矢量）
```python
比如torch.tensor([7,7])  
```
### 2.2.1 ndim  
ndim为1
### 2.2.2 shape
shape为2，为向量里面的元素个数  

## 2.3 矩阵Matrix
```python
比如MATIX = torch.tensor([[7, 8]  
                        [9, 10]])
```
### 2.3.1 MATIX.ndim 为2  
### 2.3.2 MATIX[0]
torch.tensor([7, 8])  
### 2.3.3 MATIX.shape 
torch.Size([2, 2])  

## 2.4 张量 tensor
```python
TENSOR = torch.tensor(  
    [[[1, 2, 3],  
    [3, 6, 9],  
    [2, 4, 5]]]  
)
```
### 2.4.1 TENSOR.ndim = 3
### 2.4.2 TENSOR.shape = torch.Size(1,3,3)
![alt text](image.png)
### 2.4.3 random tensors
torch.rand()  
对于图片来讲，可以表示为3个通道，每个通道为 

torch.tensor([height, width])  
torch.tensor(height, width)  
torch.tensor(size=(height, width))



### 2.4.4 全部是0或者全部是1的张量   
```python
zero = torch.zeros(size=(3, 4))  
```
清除一个张量直接tensor*zero就把原来的tensor清除了，就都是0了  
```python
ones = torch.ones(size(3, 4))
```
数据类型dtype，
```python   
ones.dtype = torch.float32
```
# 3. 创建range
## 3.1. troch.arange()  
torch.arange(start, end ,step)和原生range一样
```python
one_to_four = torch.arange(0, 4)
# 会得到
tensor([0, 1, 2, 3])
```
形成一个维度一样的全0矩阵
```python
torch.zeros_like(input=one_to_four)
# 会得到
tensor([0, 0, 0, 0])
```

## 3.2. 数据类型  
常见的pytorch中的错误：  
tensor类型错误，使用.type  
tensor形状错误，使用.size()或.shape  
tensor没在正确的device上，使用.device    

创建张量预选有几个选项比较重要， 分别是dtype, device, requires_grade
```python
float_32_tensor = torch.tensor([3.0, 6.0, 9.0]，
                                dtype=None,
                                device=None,
                                requires_grad = False)
#改变数据类型
float_16_tensor = float_32_tensor.type(torch.float16)                            
```
### 3.2.1. dtype
默认为 torch.float32, 可以强制更改为torch.float16等
.dtype来看数据类型

### 3.2.2. device
device可以是cpu，也可以是cuda，默认是None
```python
# 可以用
tensor.device()获取
```

### 3.3. 计算tensor

3.3.1. addition
创建一个tensor
```python
tensor = torch.tensor([1, 2, 3])
# 执行加标量运算
tensor + 10
# 结果为
tensor([11, 12, 13])

# 内置加法
torch.add(tensor, 10)
# 结果为
tensor([11, 12, 13])

```

3.3.2. subtraction
```python
# 执行减标量运算
tensor - 10
# 结果为
tensor([-9, -8, -7])
```
3.3.3. multiplication(element-wise)元素乘法
```python
# 执行乘法
tensor * 10
# 结果为
tensor([10, 20, 30])

# 也可以使用内置乘法
torch.mul(tensor, 10)
# 结果为
tensor([10, 20, 30])
```

3.3.4. 除法
3.3.5. 矩阵乘法
```python
# 执行矩阵乘法运算
torch.matmul(tensor, tenosr)
# 结果为
14
# 或者使用@
tensor @ tensor
# 结果为
14
# 或者使用torch.mm
torch.mm(tensor, tensor)
# 结果为
14
```
发现个很有意思的讲矩阵的网站http://matrixmultiplication.xyz/  

## 3.4. 转置
tensor.T为tensor的转置  
