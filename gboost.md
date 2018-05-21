XGBOOST算法有广泛的应用，本文主要是该算法的学习记录。

Boosting方法是集成学习中重要的一种方法，Boosting方法中，最终的预测结果为b个学习器结果的合并。
基本思想：
每个分类器对之前所有分类器的结果和正确分类的残差进行学习，求解损失函数关于的泰勒展开，将问题转化为二次函数优化
问题，求解出最值和取最值时的权重值。
具体推导如下：
分类器：
$y_i = y_{i-1}+f_t(x_i)+\Omega(f_i) $
损失函数：
$\sum_{i=1}^{n} l(y_i,y_t)+\Omega(f_t) $
现在需要求解使L取最值的f_t(x_i)
函数进行二阶泰勒展开如下
f(x+\bigtriangleup x) = f(x)+f'(x)\bigtriangleup x + \frac{1}{2} f''(x)\bigtriangleup  x^2
令g_i = l'(y_i,y_t),h_i = l''(y_i,y_t)
可得
$\sum_{i=1}^{n} {l+g_if_t(x_i)+\frac{1}{2}f^2_t(x_i) }$

定义正则化项：
\Omega(f_t) = \gamma T +\frac{1}{2}\lambda \sum_{T}^{j=1}w^2_j
正则化项可以使w_j和树的节点数有一个变小的趋势，该处为L2正则加上树的复杂程度
利用f^2_t(x_i)=1化简可以得到
\sum_{T}^{j=1}[(\sum _{i\in I_j}g_i) w_j+\frac{1}{2}(\sum _{i\in I_j}h_i+\lambda ) w^2_j]+\gamma T
此处是将原来n个nn上求和转化为叶子节点上的样本集求和,与叶子节点上的权重w^2_j结合
求解的参数：
对每个分类器的权重函数f(x)=wq(x);其中q(x)为该树的复杂程度，w为分类的权重

对比Adaboost模型，
该算法有以下步骤：
