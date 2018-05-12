---
title: tensorflow卷积函数
date: 2018-04-01 16:36:34
tags: tensorflow
mathjax: true
---

卷积是卷积神经网络CNN中的重要操作，下面是TensorFlow中卷积函数的用法。

<!-- more -->

tensorflow卷积函数：

tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)

input为输入的数组，假设维度为[batch,矩阵高x,矩阵宽y,矩阵深度z]

filterw为卷积核，维度[矩阵高x,矩阵宽y,矩阵深度z, out_channels]

strides：卷积时在图像每一维的步长

padding：是否补全空白”SAME”或”VALID”

```
#定义全1的矩阵，维数[3,3,1]
x1 = tf.constant(1.0, shape=[1,6,6,2])  
#定义核函数，维度[1,1,1]
#第一通道	
#	1	1
#	1	1
#	1	1
#第二通道	
#	1	1
#	1	1
#	1	1
kernel = tf.constant(1.0, shape=[3,2,2,1])   
y2 = tf.nn.conv2d(x1, kernel,strides=[1,1,1,1],padding='VALID')  
with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    #显示y2
    print(sess.run(tf.shape(y2))
#行6-3+1，列6-2+1
OUT:[1 4 5 1]
```

