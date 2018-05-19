---
title: python的pickle模块
date: 2018-04-28 23:01:12
tags:
---

python的pickle模块可以存储python数据文件，本文是该模块的学习记录。

<!-- more -->

#### **pickle常用方法**

##### 序列化和反序列化：

序列化就是把计算得到的数据保存起来，当需要使用时反序列化把数据恢复，这样有如下好处：

1. 被pickle的数据，在被多次reload时，不需要重新去计算得到这些数据，这样节省计算机资源，如果你不pickle，你每调用一次数据，就要计算一次。
2. 通过pickle的数据，被reload时，可以更好的被内存调用，不需要经过数据格式的转换。

pickle.dump(obj,file,protocol,)

将obj保存为文件

pickle.load(file)

读取pickle文件

```
import pickle  
t1=['a',3,4]
with open('temp.pkl','wb+') as f: 
	v1 = pickle.dump(t1)
#可以看到文件夹里多了temp.pkl
with open('d:temp.pkl','rb') as f: 
    v2=pickle.load(f)
    print(v2)
```

pickle.dumps(obj)

将obj保存为pickle对象

pickle.loads(obj)

读取pickle对象

```
# dumps功能
import pickle
data = [3,6,8]  
# dumps 将数据通过特殊的形式转换为只有python语言认识的字符串
v2 = pickle.dumps(data)
print(v2)            
mes = pickle.loads(v2)
print(mes)
```

