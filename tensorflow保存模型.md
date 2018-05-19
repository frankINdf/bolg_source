---
title: tensorflow保存模型
date: 2018-05-08 22:27:50
tags:
---



tensorflow模型可以保存为meta文件或者pb文件本文主要是这两种方式的实践。

<!-- more -->

保存为meta后，文件夹会出现4个文件：

`checkpoint`文件保存目录下所有的模型文件列表

`model.ckpt.meta`文件保存了TensorFlow计算图的结构

`model.ckpt`文件保存了TensorFlow程序中每一个变量的取值

**保存为meta文件**

模型保存：

使用`tf.train.Saver()`

在构建图时定义`saver = tf.train.Saver()`

运行后使用`saver.save(sess ,'d:/MMNIST/model2.ckpt')`将模型保存，此时文件夹中会出现上述的文件。

模型加载

tensorflow可以直接加载持久化模型

使用`saver = tf.train.import_meta_graph('D:/MMNIST/model2.ckpt.meta')`导入持久化模型，此时导入的是`meta`文件

使用`saver.restore(sess, tf.train.latest_checkpoint('D:/MMNIST/'))`加载模型

`reader = tf.train.NewCheckpointReader('D:/MMNIST/model2.ckpt') `可以读取ckpt文件中的变量

使用`get_variable_to_shape_map()`获得计算图中的所有变量，可以通过它打印出变量名

**注意**：

保存模型的路径不能含有中文

保存完成之后如果再次保存，需要将旧模型删除

**保存为pb文件**



PB 文件是表示 MetaGraph 的 protocol buffer格式的文件，MetaGraph 包括计算图，数据流，以及相关的变量和输入输出signature以及 asserts 指创建计算图时额外的文件。



使用`tf.SavedModelBuilder `类来完成这个工作，并且可以把多个计算图保存到一个 PB 文件中，如果有多个MetaGraph，只会保留第一个 MetaGraph 的版本号，并且必须为每个MetaGraph 指定特殊的名称 tag 用以区分，通常这个名称 tag 以该计算图的功能和使用到的设备命名，比如 serving or training， CPU or GPU。

模型保存：

使用`tensorflow.python.framework.convert_variables_to_constants(sess,sess.graph_def,['op_to_store'])`将模型持久化

使用`tf.gfile.FastGFile(path,mode)`创建pb文件

使用f.write(contant_graph.SerializeToString())将文件写入

可以看到制定文件夹中会增加**.pb文件

模型读取：

`with gfile.FastGFile('D:/MMNIST/modelpb.pb', 'rb') as f:`打开pb文件

`graph_def.ParseFromString(f.read())`读取文件f中的内容

`tf.import_graph_def(graph_def)`导入计算图

导入计算图后进行变量初始化`sess.run(tf.global_variables_initializer())`

此时模型已经导入到当前计算图，可以读取保存的计算图中的内容或者传入placeholder计算结果

保存为meta文件代码

```
from __future__ import print_function
import os
# Import MNIST data
from tensorflow.python.framework import graph_util
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
logs_path = os.getcwd()
import tensorflow as tf
tf.reset_default_graph()
# Parameters
learning_rate = 0.1
num_steps = 10
batch_size = 128
display_step = 20

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1]),name = 'h1_w'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name = 'h2_w'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]),name = 'out_w')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]),name = 'b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]),name = 'b2'),
    'out': tf.Variable(tf.random_normal([num_classes]),name = 'out')
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    # Run the initializer
    sess.run(tf.global_variables_initializer())
    constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def)    
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        _,c=sess.run([train_op,loss_op], feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    saver.save(sess,'D:/MMNIST/model2.ckpt.meta')
    print("Optimization Finished!")
```

读取meta文件的代码

```
import os
import tensorflow  as tf
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow
tf.reset_default_graph()
meta_file_path = os.getcwd()

sess = tf.Session()
saver = tf.train.import_meta_graph('D:/MMNIST/model2.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('D:/MMNIST/'))
reader = tf.train.NewCheckpointReader('D:/MMNIST/model2.ckpt')  
  
variables = reader.get_variable_to_shape_map()  
  
for ele in variables:  
    print(ele)  
```

保存pb文件

```


from __future__ import print_function
import os
# Import MNIST data
from tensorflow.python.framework import graph_util
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
logs_path = os.getcwd()
import tensorflow as tf
tf.reset_default_graph()
# Parameters
learning_rate = 0.1
num_steps = 10
batch_size = 128
display_step = 20

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input],name = 'x')
Y = tf.placeholder("float", [None, num_classes], name = 'y')

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1]),name = 'h1_w'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name = 'h2_w'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]),name = 'out_w')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]),name = 'b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]),name = 'b2'),
    'out': tf.Variable(tf.random_normal([num_classes]),name = 'out')
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name = 'op_to_store')

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,['op_to_store'])    
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        _,c=sess.run([train_op,loss_op], feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    with tf.gfile.FastGFile('D:/MMNIST/modelpb.pb',mode = 'wb') as f:
        f.write(constant_graph.SerializeToString())
    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))

```

读取pb文件

```
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
sess = tf.Session()
with gfile.FastGFile('D:/MMNIST/modelpb.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def) # 导入计算图

# 初始化的过程    
sess.run(tf.global_variables_initializer())

# 需要先复原变量
print(sess.run('b1:0'))
## 1
#
## 输入
input_x = sess.graph.get_tensor_by_name('x:0')
print(input_x)
#input_y = sess.graph.get_tensor_by_name('y:0')
#
op = sess.graph.get_tensor_by_name('op_to_store:0')
print(op)
```

