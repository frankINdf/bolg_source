## VISUALIZING AND UNDERSTANDING RECURRENT
NETWORKS ##

__从三个方面解释LSTM__

1) 内部结构

2) 记忆能力

3) 错误原因

### Long Short-Term Memory ###

解决的问题：RNN的梯度爆炸和消失

一般梯度爆炸借鉴梯度截断的处理方式

LSTM设计出来减轻梯度消失

增加了$c_t^l$, 通过两个门读写和重置这个cell

$c_t^l = f * c_{t-1}^l + i * g$

$h_t^l = o * tanh(c_t^l)$

sigm和tanh来作为激活函数

$i, f, o$三个门用来控制memory cell的更新

$g$用来修改memory content

由于后向传播时有各个状态梯度相加

允许memory cell $c$的梯度在很长的时间步不连续的后向传播

至少直到？？？


### 实验设计 ###

#### Internal mechanisms of an LSTM ####

__Interpretable long range LSTM cells__

这段说了什么？？

实验说明：各个cell有不同的作用

例如：

有的单元与positon有关，随时间decaying

有的cell在括号里面时会激活

__gate activation statistics__

通过单元的激活与否，可以观察LSTM的内部机制

定义小于0.1为未激活，大于0.9为激活

forget gates可以控制长时间记住某些信息



第一层RNN偏向非饱和状态

### Understanding Long-Range Interactions ###

任务是：括号或引号内内容预测

Base line:

1) 全连接网络+tanh激活, 输入是one-hot维度是nK, 交叉验证

2) n-gram模型，使用Kneser-Ney平滑

__表现:__

n变大时n-NN模型逐渐出现过拟合，n-gram表现好一点

计算单词中每个字符平均概率


### 基本RNN ###

可视化方式1： $h_t$的激活情况

可视化方式2: 隐藏层的梯度， 通过观察梯度改变的的大小

### embedding ###

观察训练前后的embedding

### LSTM 可视化 ###
