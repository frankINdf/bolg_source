---
title: numpy的argsort函数用法
date: 2018-03-28 22:17:28
---

数组操作时有时如果需要获取数组的索引可以使用argsort函数。

<!-- more -->

numpy中的argsort可以返回一个索引，具体参数如下

numpy.argsort(a, axis=-1, kind='quicksort', order=None)

a数组

axis行或者列，默认为-1，即最后一个维度，0为列，1为行

kind排序方式{‘quicksort’, ‘mergesort’, ‘heapsort’}

order

返回的是排序后的数组在原数组中索引的数组。

```
import numpy as np
a=np.array(([[1,6,3,4,2]]))
a.argsort()
#返回的是ARRAY从小到大的索引
Out[35]: array([[0, 4, 2, 3, 1]], dtype=int64)
#默认为行排序的索引
np.argsort(a)
Out[41]: 
array([[0, 3, 2, 1],
       [1, 0, 3, 2]], dtype=int64)
#axis为1按列排序
np.argsort(a,axis=0)
Out[39]: 
array([[0, 1, 0, 0],
       [1, 0, 1, 1]], dtype=int64)
#axis为1，按行排序
np.argsort(a,axis=1)
Out[40]: 
array([[0, 3, 2, 1],
       [1, 0, 3, 2]], dtype=int64)
```

##  