---
title: GBDT算法
date: 2018-05-21 21:36:45
tags:
---



GBDT和XGBOOST算法有广泛的应用，本文主要是该算法的学习记录。

<!-- more -->

Boosting方法是集成学习中重要的一种方法，Boosting方法中，最终的预测结果为b个学习器结果的合并。
**基本思想：**
每个分类器对之前所有分类器的结果和正确分类的残差进行学习，求解损失函数关于的泰勒展开，将问题转化为二次函数优化问题，求解出最值和取最值时的权重值。
具体推导如下：
分类器：
$y_i = y_{i-1}+f_t(x_i)+\Omega(f_i) $

$\Omega(f_i)$为正则化项。 

**目标函数**：
$\sum_{i=1}^{n} l(y_i,y_t)+\sum^{t}_{i=1} \Omega(f_i) $
现在需要求解使L取最值的$f_t(x_i)$
函数进行二阶泰勒展开如下
$f(x+\bigtriangleup x) = f(x)+f'(x)\bigtriangleup x + \frac{1}{2} f''(x)\bigtriangleup  x^2$
对于目标函数，令$g_i = l'(y_i,y_t),h_i = l''(y_i,y_t)$
可得
$\sum_{i=1}^{n} {l+g_if_t(x_i)+\frac{1}{2}f^2_t(x_i) }$

**正则化项**：

定义正则化项：
$\Omega(f_t) = \gamma T +\frac{1}{2}\lambda \sum_{T}^{j=1}w^2_j$
正则化项可以使w_j和树的节点数有一个变小的趋势，该处为L2正则加上树的复杂程度.
利用f^2_t(x_i)=1化简可以得到
$\sum_{T}^{j=1}[(\sum _{i\in I_j}g_i) w_j+\frac{1}{2}(\sum _{i\in I_j}h_i+\lambda ) w^2_j]+\gamma T$
此处是将原来所有样例乘权重求和转化为叶子节点上所有样本集乘权重求和,与叶子节点上的权重$w^2_j$结合

**优化过程**

定义$G=\sum _{i\in I_j}$,$H=\sum _{i\in I_j}h_i$

目标函数变为：$\sum_{T}^{j=1}[G_{w_j}+\frac{1}{2}(H_j+\lambda)w^2_j]+\gamma T$

当$w^*_j =-\frac{G_j}{H_j+\lambda}$时，最值为$-\frac{1}{2}\sum^{T}_{j=1}\frac{G^2_j}{H_j+\lambda}+\gamma T$

**分割点选择**

类似决策树算法，对每个点计算$Gain$,最大点即为分类点，同时可以算出该分类的权重:

$Gain = \frac{1}{2}[\frac{G^2_L}{H_L+ \lambda} +\frac{G^2_R}{H_R+\lambda}-\frac{(G2_L+G_R)^2}{H_L+H_R+\lambda}]- \gamma$

