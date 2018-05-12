---
title: tensorflow中的softmax
date: 2018-04-20 21:58:53
mathjax: true
tags:
---

softmax函数在机器学习中应用很广泛，本文主要记录TensorFlow中的交叉熵计算函数用法。

<!-- more -->

神经网络中需要将正向传播的结果和的正确结果进行进行对比，softmax函数定义如下，它将分类结果映射到[0,1]这个区间,Vi、Vj表示V中第i，j个元素，ai可以看成第i个分类结果：
$$
a_i =\frac{e^{V _i}}{\sum_je^{V_j} }
$$
交叉熵C如下，其中yi代表真实值，ai在这里为softmax：
$$
C = -\sum_i{y_i{log(a_i)}}
$$
计算两者之间的差距,每项的Loss可以用下式表示，对于**只有1个正确分类i的分类**,softmax交叉熵计算公式如下：
$$
L_i = -log(\frac{e^{f _{y_i}}}{\sum_je^{f_{y_j}} }  )
$$
可以看到，括号里即为softmax的值，它越大，样本的Loss就越小，即与真实分布的差距越小。

在TensorFlow中交叉熵有下面几种计算方法：

- tf.nn.softmax_cross_entropy_with_logits（label, logits）

  logits的shape=(m,n)，label的shape=(m,n),，如果真实分类logits为一维数组，则需要进行one-hot编码。


- tf.nn.sparse_softmax_cross_entropy_with_logits（label, logits）
- logits的shape=(m,n)，label的shape=(m,1),若label的shape=(m,n)阶则需要使用argmax函数变为(m,1)
- tf.nn.sigmoid_cross_entropy_with_logits：张量中标量与标量间的运算,求两点分布 之间的交叉熵。

```python
#下面举例进行交叉熵计算
#分别用softmax_cross_entropy_with_logits、tf.nn.sparse_softmax_cross_entropy_with_logits
#手算和tf.nn.sigmoid_cross_entropy_with_logits
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
#label代表真实分布
#采用one-hot编码，即真实分类分别为[3,2,1,1,2]
labels = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=np.float32)
#logits代表前向传播的结果
#代表每个值得权重
logits = np.array([[1, 2, 7],
                   [3, 5, 2],
                   [6, 1, 3],
                   [8, 2, 0],
                   [3, 6, 1]], dtype=np.float32)
num_classes = labels.shape[1]
#tf.nn.softmax用来求softmax值，即映射到[0,1]区间上的概率
predicts = tf.nn.softmax(logits=logits, dim=-1)
sess.run(predicts)
```

```
out:array([[  2.45611509e-03,   6.67641265e-03,   9.90867496e-01],
       [  1.14195190e-01,   8.43794703e-01,   4.20100652e-02],
       [  9.46499169e-01,   6.37746137e-03,   4.71234173e-02],
       [  9.97193694e-01,   2.47179624e-03,   3.34521203e-04],
       [  4.71234173e-02,   9.46499169e-01,   6.37746137e-03]], dtype=float32)
```

```python
#使用tf.nn.softmax_cross_entropy_with_logits计算交叉熵，labels可以直接用one-hot编码的数组
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
sess.run(cross_entropy)
```

```
out:array([ 0.00917445,  0.16984604,  0.05498521,  0.00281022,  0.05498521], dtype=float32)
```

```python
#使用tf.nn.sparse_softmax_cross_entropy_with_logits计算交叉熵，labels要处理为(m,1)维
classes = tf.argmax(labels, axis=1)
sess.run(classes)
cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=classes)
sess.run(cross_entropy2)
```

```
out:array([ 0.00917445,  0.16984604,  0.05498521,  0.00281022,  0.05498521], dtype=float32)
```

```python
#直接用reduce_sum计算，其中
#用clip_by_value将取值限制在1e-10以上，防止出log(0)
#用labels/predicts相当于前面加负号
labels = tf.clip_by_value(labels, 1e-10, 1.0)
predicts = tf.clip_by_value(predicts, 1e-10, 1.0)
cross_entropy4 = tf.reduce_sum(labels * tf.log(labels/predicts), axis=1)
sess.run(cross_entropy4)
```

```
array([ 0.00917445,  0.16984604,  0.05498521,  0.00281022,  0.05498521], dtype=float32)
```

```python
z = 0.8
x = 1.3
cross_entropy3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=x)
# tf.nn.sigmoid_cross_entropy_with_logits的具体实现:
cross_entropy5 = - z * tf.log(tf.nn.sigmoid(x))  - (1-z) * tf.log(1-tf.nn.sigmoid(x))
sess.run(cross_entropy3)   
```

```
0.50100845
```



