# Transformer 问题 #

## 什么是注意力机制 ##

encode时, 注意力机制就是找输入序列之间的关系

如果某两个词有隐藏的联系

weight会变大, 反之变小

的到weight之后, 可以加权得到新的特征向量

## 怎么得到注意力权重 ##

将输入的表征线性变换, 得到3个特征: Q, K, V

$softmax(\frac{Q*K}{\sqrt{d_k}})V$

## encode 和 decode有什么区别 ##

encode时是自注意力, Q K V都来自input的query

decode时K V来自decode的input

## encode 和 decode是怎么连接的 ##

encode最后一层是decode所有层的输入

也就是说, 在encode激活的部分, 会影响decode的每一步

## 其他要点 ##

position embedding: 添加了位置信息

feed forward: 使用全连接, batch normal, Relu等函数

残差连接: 每层的V和下一层输出的相加

mulit head: 在self attention之前, 将特征向量截取为多个, 每段进行计算, 最后的多段进行拼接

mulit head 有非线性的作用, 也能减小计算量



## 如何做翻译、做分类 ##

翻译的时候, 看过了整句, 所以对应位置输出为全局最合理

__会用到上一步的输入么__:不会, 和LSTM不同, 不用将上一步的向量输入


__对结果进行搜索时, 怎么优化__



