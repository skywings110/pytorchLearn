# pytorch Learn  

1. 环境配置  
1.1 使用colab.research.google.com真的好用，赞！  
1.2 ctrl+m m转换代码为文本  
1.3  导包几件套  
import torch  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

2. 常用命令  
2.1 标量，比如torch.tensor(2)  
2.1.1 维度ndim  
scalar = torch.tensor().ndim，标量没有维度  
2.1.2 获取原本数值  
torch.tensor().item()  
2.2 向量（矢量），比如torch.tensor([7,7])  
2.2.1 ndim  
ndim为1
2.2.2 shape
shape为2，为向量里面的元素个数  
2