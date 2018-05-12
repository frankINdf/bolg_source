---
title: SVM支持向量机
date: 2018-04-21 18:46:53
mathjax: true
tags:

---

SVM是经典机器学习方法，本文主要是对SVM原理、求解等方面的理解。

<!-- more -->

SVM分类使用sigmoid函数判断结果。

**几何角度的解释**：不同分类之边界之间间隔最大化
$$
 { \ min _ { w , b } } \frac{1}{2}\left \| w \right \|^2
$$

$$
s . t .  y_i(w^Tx_i+b)\geqslant 1 , i=1,2...,m.
$$

**损失函数最小解释**：$1/2||w||^2$为结构化风险的L2正则项，损失函数为合页函数$l_{0/1}$为$max(0,f(x))$，
$$
{\min_{w,b}}\frac{1}{2}\left \| w \right \|^2 + C\sum_{i=1}^{m}\ l _{0/1}(y_i(w^Tx_i+b)-1)
$$

$$
s.t     (y_i(w^Tx_i+b)-1)\leqslant 1-\xi _i
$$

$$
\xi _ i \geqslant 0
$$

采用拉格朗日法：
$$
L(w,b,\xi,\alpha,\mu) = \frac{1}{2}\left \| w \right \|^2 + C\sum_{i=1}^{m}\ l _{0/1}+\sum_{m}^{i=1}\alpha_i[y_i(w^Tx_i+b-1+l_{0/1})]-\sum_{m}^{i=1}{\mu_i\xi_i}
$$
依次对$w,b,\xi_i$求偏导，得到
$$
\min_{w,b,\xi}L = \sum_{i=1}^{N}a_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N}a_i a_j y_i y_jK(x_i,x_j)
$$
原始问题满足KKT条件，对偶问题变为：
$$
\max \sum_{i=1}^{N}a_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N}a_i a_j y_i y_jK(x_i,x_j)
$$

$$
s.t.\space\sum_{j=1}^{N}a_iy_i=0
$$

$$
0 \leqslant a_iy_i \leqslant C
$$

**惩罚因子**C: C代表对损失函数的惩罚程度，C越大损失函数对优化影响越大，损失函数取的系数C正无穷时，不允许有异常点，即为硬间隔。

**损失函数**： $l_{0/1}$理解为$y_i(w^Tx_i+b)-1>0$时无损失，无需继续优化，这里可以看到$l_{0/1}$是“没有追求”的，只要预测正确就不再努力使预测值向正方向增加，这也使SVM算法可以对异常值有一定“容忍”。

**核函数：**把低维度的数据映射到高维度，用空间变换找数据的分割面，找到分割面后再转换到原空间，即可画出分界边界，例如$x^2+y^2+b$ 在新空间$\alpha+\beta+b$为直线，在原空间线性不可分的点在新空间里线性可分。

RBF核$K(x,x_i)=exp(\frac{||x-x_i||^2}{σ^2})$会将原始空间映射为无穷维空间,$σ$ 选得大，高次特征上的权衰减变快， *σ* 选得小，可以将任意的数据映射为线性可分，可能过拟合。**高斯核实际上具有相当高的灵活性，也是使用最广泛的核函数之一。**

**SMO算法：**简单理解SMO就是选定$a_i,a_j$将其他项看做常数，可以用$a_i$表示$a_j$,再在[0,C]这个区间求解极值，依次循环。

**注意**

SVM没有处理缺失值的策略，使用SVM前需要将数据进行处理

例1：SVM示例

绘制SVM分类示意图，用clf.coef_和clf.intercept获得w和b，用clf.support_vectors_获得支持向量

```
# coding: utf-8
import numpy as np
from sklearn import svm
import pylab as pl
np.random.seed(0) # 使用相同的seed()值，则每次生成的随即数都相同
# 创建可线性分类的数据集与结果集
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20,2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# 构造 SVM 模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y) # 训练 
#利用clf.coef_得到系数w
#clf.intercept_[0]是b
xx = np.linspace(-5, 5) # 在区间[-5, 5] 中产生连续的值，用于画线
yy =  -w[0] * xx / w[1]  - (clf.intercept_[0]) / w[1]
b = clf.support_vectors_[0] # 第一个分类的支持向量，通过调整系数b
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1] # 第二个分类中的支持向量
yy_up = a * xx + (b[1] - a * b[0])

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')
pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
```

![](D:\blog\source\_posts\SVM支持向量机\下载.png)

