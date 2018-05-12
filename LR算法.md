---
title: LR算法
date: 2018-04-22 20:36:45
mathjax: true
tags:
---

本文主要分析逻辑回归的原理和自己的一点思考。

<!-- more -->

算法原理：

选取sigmoid函数作为logistic的概率分布函数
$$
P(Y=1|x) = \frac{\mathrm{exp}(wx+b)}{1+\mathrm{exp}(wx+b)}
$$

$$
P(Y=0|x) = \frac{1}{1+\mathrm{exp}(wx+b)}
$$

用线性模型$$wx+b$$逼近对数几率$$log \frac{y}{1-y}$$即
$$
log\frac{P(Y=1|x)}{P(Y=0|x)} = wx+b
$$
对于0-1二分类，概率为p和1-p,符合伯努利分布，参数的似然函数为：
$$
L(W)=P(D|p_{yi})=\prod p^{yi}(1-p)^{1-yi}
$$
取对数
$$
logL = \sum  [ y_ilogp + (1-y_i)log(1-p) ]
$$

$$
logL = \sum [ y_ilog \frac{p}{1-p} +log(1-p) ]
$$

$$
logL = \sum [ y_i(w_ix+b) +\mathrm log(1+e^{w_ix+b)} 
]
$$

采用梯度下降法和牛顿法可以求解，下面介绍梯度下降法。

对w求偏导,其中：
$$
\frac{\partial logL}{\partial w_i} = \sum x_i(yi-p_i)
$$
得到梯度后就可以迭代下个w
$$
w_{new}  =  w_{old} + \alpha \frac{\partial logL}{\partial w_i}
$$
也可以从损失函数的角度理解，$$y_i=1$$和$$y_i = 0$$时对数损失函数$$log(p(y|x))$$如下（将yi为0和1带入对数内部）：


$$
cost  ( h  _ \theta (x) , y )  =  \left\{\begin {matrix} -y_ilog(h_ \theta(x)) \! \: \: \: \:  if\:  y = 1\\ -(1-y_i)log(1-h_ \theta(x)) \! \: \: \: \:  if\:  y = 0\end{matrix}\right.
$$


联合起来，用一个式子表示：
$$
cost(h_ \theta(x),y ) = -y_ilog(h_ \theta(x)) -(1-y_i)log(1-h_ \theta(x))
$$

$$
cost(h_ \theta(x),y ) =  -\frac{1}{2}\sum (y_ilog(h_ \theta(x)) +(1-y_i)log(1-h_ \theta(x)))
$$

接下来计算和之前一样。

**LR损失函数的形式，其实就是交叉熵。**

**损失函数：**
LR使用对数损失对于sigmoid函数损失函数也可以表示成：
$$
L(y_i(wx+b)) = log_2(1+e^{y_i(wx+b)})
$$
损失函数的图像如下：

![失函数曲](LR算法\损失函数曲线.png)

**损失函数的意义:**

损失函数的值总大于0-1分段函数，这保证了求解的准确性，使损失函数向正方向变化。可以从损失函数看到，即使预测值与实际值完全一样也就是横坐标大于0，损失函数还是不为0，说明有一个正方向的梯度，这就使LR即使全部预测值分类正确，但还是要优化到分界曲线的位置，如果有异常值，也会将其考虑进去，这点和SVM不同。

**优点**:速度快

**一点思考：**
sigmoid函数图像接近分段函数，但是这个函数有个缺点，在两端导数很小，所以用梯度下降法时，偏离正确值时可能出现迭代太慢，但是对于sigmoid函数y' = y(1-y)，对损失函数求导后与sigmoid的导数无关，函数的特性使其方便计算，如果用的是欧式距离，求导会出现y'，对迭代求解不利。



