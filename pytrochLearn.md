

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

## 3.3. 计算tensor

### 3.3.1. addition
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

### 3.3.2. subtraction
```python
# 执行减标量运算
tensor - 10
# 结果为
tensor([-9, -8, -7])
```
### 3.3.3. multiplication(element-wise)元素乘法
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

### 3.3.4. 除法
### 3.3.5. 矩阵乘法
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
## 3.5. min, max, mean, sum等
针对向量
```python
x = torch.arange(0, 100, 10)
```
### 3.5.1. find min
```python
torch.min(x)
# 或者
x.min()
```
### 3.5.2. find max
```python
torch.max(x)
# 或者
x.max()
```
### 3.5.3. find mean
mean不能处理int类型向量
```python
torch.mean(x.type(torch.float32))
# 或者
x.type(torch.float32).mean()
```
### 3.5.4. find the sum
```python
torch.sum(x), x.sum()
```
### 3.5.5. find the positional min and max
```python
# find min
x.argmin()
# result
tensor(0)
# find max
x.argmax()
# result
tensor(9)
```
# 4. 改变形状
```python
# create tensor
x = torch.arange(1., 11.)
# result
tensor([1., 2., ... 10.])
# shape
x.shape
# result
torch.Size([10])
```
## 4.1. reshaping
维度必须一致
```python
x.reshape(10,1)
x.shape
# result
tensor([[1.],
        [2.],
        ...
        [10.]]),
torch.Size([10,1])       
# another reshape
x.reshape([5, 2])
# result
tensor([[1.],[2.],
        ...
        [1.],[10.]]),
```
## 4.2. view
调整形状，共享相同内存的，改变z就会改变x，纯纯浅拷贝的感觉
```python
x = torch.arange(1., 10.)
x, x.shape
tensor([1., 2., ... , 9.])
torch.Size(9)
# result
z = x.view(1, 9)
z, z.shape
# result
tensor([[1., 2., ... , 9.]])
torch.Size([1, 9])
```
## 4.2. stack
把一堆向量堆成一个矩阵，dim是堆叠方向
```python
x_stacked = torch.stack([x, x, x, x], dim = 0)
# result
tensor([[1., 2., ... 8., 9.],
        [1., 2., ... 8., 9.],
        [1., 2., ... 8., 9.],
        [1., 2., ... 8., 9.]])
# dim=1的堆叠方式
x_stacked = torch.stack([x, x, x, x], dim = 1)
# result
tensor([[1., 1., 1., 1.],
        [2., 2., 2., 2.],
        ...,
        [9., 9., 9., 9.]])
```
## 4.3. squeezing and unsqueezing tensors.
squeeze把一个多维的矩阵挤成一个向量，unsqueeze 增加维度
```python
x = torch.arange(1., 10.)
z = x.view(1, 9)
z, z.shape
# result
tensor([[1., 2., ... , 9.]])
torch.Size([1, 9])
# squeeze一下
z.squeeze(), z.squeeze().shape
# result
tensor([1., 2., ... , 9.])
torch.Size([9])
# unsqueeze一下，dim=0
z_unsqueezed = z.squeeze().unsqueeze(dim=0)
# result
tensor([[1., 2., ... , 9.]])
# shape
tensor([1, 9])
# unsqueeze一下，dim=1
z_unsqueezed = z.squeeze().unsqueeze(dim=1)
# result
tensor([[1.], 
        [2.], 
        ... , 
        [9.]])
# shape
tensor([9, 1])
```
## 4.4. permute
permute的功能是交换维度，与view一样，改变里面数据会改变原来的数据。
```python
x_orginal = torch.rand(size(224, 224, 3))
# x_orginal是一个三维的张量
# 进行permute变换
x_permuted = x_original.permute(2, 0, 1)
# 这个permute代表轴从2->0, 0->1, 1->2, 轴从0开始算。
```
## 4.5. 索引
跟numpy一样

## 4.6. 跟numpy的转换
主要说一下数据类型，from_numpy()和numpy()的用法，如果直接转换，tensor的arange是32位，numpy的arange是64位，如果强制转换的话，就from_numpy().type('float32')
### 4.6.1. numpy to tensor
```python
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
# 数据类型
array.dtype
# result
dtype('float64')
# tensor.dtype
tensor.dtype
# result
dtype('torch.float64')
# 如果是直接用torch.arange的数据，就是float32
torch.arange(1.0, 8.0).dtype
# result
torch.float32
```
### 4.6.2 tensor to numpy
同上，其实没啥

## 4.7. reproducbility(从随即中剔除随机)
即选种子
```python
# 纯随机
torch.rand(3,3)
# 使用种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM)
random_tensor_C = torch.rand(3, 4)
random_tensor_D = torch.rand(3, 4)
# 这样两个张量就完全一致
print(random_tensor_C == random_tensor_D )
# result
tensor([[True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]])
```
# 5. GPU加速
## 5.1. 使用GPU
```python
# 查询GPU命令
!nividia-smi
# 检查GPU是否可用
torch.cuda.is_available()
# 设置使用GPU
device = "cuda"
# 查询GPU设备数量，离谱，有一个就不错了
torch.cuda.device_count()
```
## 5.2. 在GPU上使用tensor
```python

```