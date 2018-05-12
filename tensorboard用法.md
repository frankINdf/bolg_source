---
title: tensorboard应用
date:  2018-04-01 17:28:48
tags:  tensorflow
mathjax: true
---
通过TensorBoard可以对神经网络结构和参数收敛有更深刻的理解，本文记录了TensorBoard的应用方法。

<!-- more -->

标量数据汇总和记录使用

tf.summary.scalar(tags, values, collections=None, name=None)  

直接记录变量var的直方图

tf.summary.histogram(tag, values, collections=None, name=None）  

输出带图像的probuf，汇总数据的图像的的形式如下： ' *tag* /image/0', ' *tag* /image/1', etc.，如：input/image/0等

tf.summary.image(tag, tensor, max_images=3, collections=None, name=None)  

汇总再进行一次合并

tf.summary.merge(inputs, collections=None, name=None)

合并默认图形中的所有汇总

tf.summaries.merge_all(key='summaries')  

下面看一个TensorBoard的例子：

```
'''
通过
'''
from __future__ import print_function
import tensorflow as tf
#输入数据，这里用的是mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#学习率
learning_rate = 0.01
#训练集的训练次数
training_epochs = 25
#每次输入的数据个数
batch_size = 100
display_epoch = 1
#数据存储位置
logs_path = '/tmp/tensorflow_logs/example/'
# mnist数据集是28*28的图片，一共784个像素
#训练集x
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
#真实值y，对数字分类有10个类别
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

#定义神经网络参数y=wx+b中的x和b
W = tf.Variable(tf.zeros([784, 10]), name='Weights')
b = tf.Variable(tf.zeros([10]), name='Bias')

# 在命名空间内定义模型、损失、优化方法
with tf.name_scope('Model'):
    # 训练模型，采用全连接神经网络
    #matmul进行矩阵相乘
    #softmax得到输出结果
    pred = tf.nn.softmax(tf.matmul(x, W) + b) 
with tf.name_scope('Loss'):
	#损失函数使用交叉熵，这里也可以用TF自带的函数
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
with tf.name_scope('SGD'):
    #用梯度下降法对损失函数进行优化
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    # 计算准确率，这里argmax的参数就对应分类
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# 计算之前需要将全局变量初始化
init = tf.global_variables_initializer()
# 添加变量到TensorBoard
# 这里添加了损失、准确率，在输出的文件中可以看到这些参数的变化情况
tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:

    # 初始化数据
    sess.run(init)

    # 初始化TB，路径为logs_path
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # 开始训练过程
    for epoch in range(training_epochs):
        avg_cost = 0.
        #所有数据训练一次要分成total_batch个数据集
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 训练数据batch_xs,batch_ys
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # 将数据写入之前初始化的TB中
            summary_writer.add_summary(summary, epoch * total_batch + i)
            #计算训练一次的平均损失函数
            avg_cost += c / total_batch
        # 显示每个Epoch的cost
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # 测试集上显示准确率
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")

```

运行完程序后，输出如下：

```
[out]：Epoch: 0001 cost= 1.183585649
Epoch: 0002 cost= 0.665355021
Epoch: 0003 cost= 0.552772734
Epoch: 0004 cost= 0.498669290
Epoch: 0005 cost= 0.465467163
Epoch: 0006 cost= 0.442601420
Epoch: 0007 cost= 0.425460528
Epoch: 0008 cost= 0.412206892
Epoch: 0009 cost= 0.401397175
Epoch: 0010 cost= 0.392420013
Epoch: 0011 cost= 0.384768393
Epoch: 0012 cost= 0.378172010
Epoch: 0013 cost= 0.372432202
Epoch: 0014 cost= 0.367334918
Epoch: 0015 cost= 0.362694857
Epoch: 0016 cost= 0.358602089
Epoch: 0017 cost= 0.354879144
Epoch: 0018 cost= 0.351492124
Epoch: 0019 cost= 0.348284597
Epoch: 0020 cost= 0.345425291
Epoch: 0021 cost= 0.342768804
Epoch: 0022 cost= 0.340257174
Epoch: 0023 cost= 0.337940815
Epoch: 0024 cost= 0.335757064
Epoch: 0025 cost= 0.333699012
Optimization Finished!
Accuracy: 0.9135
```

运行tensorboard --logdir=/tmp/tensorflow_logs，打开浏览器进入http://localhost:6006查看结果。

​    ![1](tensorboard用法\1.png)

‘    ![2](tensorboard用法\2.png)





