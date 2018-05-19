---
title: tensorflow中的滑动平均值
date: 2018-04-28 20:10:50
tags:
---



在tensorflow中使用滑动平均值可以有效提高计算效率，本文为滑动平均值的学习记录。

<!-- more -->

**计算原理**

TensorFlow中采用`tf.train.ExponentialMovingAverage`函数更新参数，函数每次更新参数后需要和上一轮的变量按照下面公式进行更新：

$new\_value = (1-\alpha)*value+\alpha *old\_ value$




可以看到参数更新的本质就是采用一阶低通滤波法对本次采样值与上次滤波输出值进行加权，得到有效滤波值，使得输出对输入有反馈作用，同时参数波动会减小，$\alpha​$越大参数更新越慢，$\alpha​$越小参数越灵敏。



**直接定义滑动平均值对象**

1. 定义需要更新的参数v、本次更新为第几步的参数step(可选)
2. 定义滑动平均值`ema = tf.train.ExponentialMovingAverage(decay,step)`
3. 将需要更新的参数v添加到ema，使用`ema,apply([v])`
4. `sess.run`更新滑动平均值
5. 重复1-4

```
var0 = tf.Variable(...)
var1 = tf.Variable(...)

opt_op = opt.minimize(my_loss, [var0, var1])

# 
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

with tf.control_dependencies([opt_op]):
    # Create the shadow variables, and add ops to maintain moving averages
    # of var0 and var1. This also creates an op that will update the moving
    # averages after each training step.  This is what we will use in place
    # of the usual training op.
    training_op = ema.apply([var0, var1])
    
    
    
```

**定义变量将其保存为滑动平均值**

1.定义滑动平均值对象

2.定义滑动平均值变量名

3.将变量保存到对应滑动平均值名

```
# 定义滑动平均值变量名shadow_var0_name，shadow_var1_name，将var0和var1保存对应这两个滑动平均值
# 加载后var0和var1以滑动平均的方式计算
shadow_var0_name = ema.average_name(var0)
shadow_var1_name = ema.average_name(var1)
saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
saver.restore(...checkpoint filename...)
```



**函数的使用**

`tf.train.ExponentialMovingAverage`定义滑动平均值计算，定义一个滑动平均值计算对象

初始化参数

```
(   decay,  #衰减率
    num_updates=None, #更新的步数，由函数自己维护
    zero_debias=False, #
    name='ExponentialMovingAverage' #滑动平均值的名称
    )
```

在tensorflow中，为了保证开始时参数更新够快,实际的更新公式如下：

$min(decay, (1 + num_updates) / (10 + num_updates))$

**apply（var_list）**

将需要更新的参数赋给计算函数，使用`trainable=False`可以创建影子变量,通过`tf.global_variables()`可以返回step.

- ​

```
__init__(
    decay,
    num_updates=None,
    zero_debias=False,
    name='ExponentialMovingAverage'
)
```



**average(var)**

使用average方法返回滑动平均值

**average_name(var)**

返回滑动平均值的变量名

**计算滑动平均值实践**

```
import tensorflow as tf
sess = tf.InteractiveSession()
v1 = tf.Variable(0, dtype=tf.float32)   # 定义一个变量，初始值为0
step = tf.Variable(0, trainable=False)    # step为迭代轮数变量，控制衰减率
ema = tf.train.ExponentialMovingAverage(0.5)  # 初始设定衰减率为0.99
maintain_averages_op = ema.apply([v1])                 # 更新列表中的变量
init_op = tf.global_variables_initializer()        # 初始化所有变量
sess.run(init_op)
print(sess.run([v1, ema.average(v1)]))                # 输出初始化后变量v1的值和v1的滑动平均值
sess.run(tf.assign(v1, 5))                            # 更新v1的值
sess.run(maintain_averages_op)                        # 更新v1的滑动平均值
print(sess.run([v1, ema.average(v1)]))
sess.run(tf.assign(step, 10000))                      # 更新迭代轮转数step
sess.run(tf.assign(v1, 10))
sess.run(maintain_averages_op)
print(sess.run([v1, ema.average(v1)]))
                                                      # 再次更新滑动平均值，
sess.run(maintain_averages_op)
print(sess.run([v1, ema.average(v1)]))
                                                      # 更新v1的值为15
sess.run(tf.assign(v1, 15))

sess.run(maintain_averages_op)
print(sess.run([v1, ema.average(v1)]))
```

```
out：
[0.0, 0.0]
[5.0, 2.5]#更新值为5，decay=0.5,结果=(5*0.5+0*0.5)
[10.0, 6.25]#更新值为10，结果=(2.5*0.5+10*0.5)
[10.0, 8.125]
[15.0, 11.5625]
```

