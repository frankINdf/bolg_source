---
title: 能生成文本的RNN
date: 2018-04-29 21:54:53
tags:
---

本文是一个RNN文本生成器的实践。

<!-- more -->

首先定义一些工具函数，这些函数可以将文本生成字典，并实现向量化。

神经网络在进行计算时，样本过大，整个训练集同时训练需要巨大的资源，因此需要将其分为数个Batch进行批训练。`batch_generator`函数用来生成Batch。函数流程如下：

复制数据 -> 计算每批参数大小 -> 计算分批数 ->  改变arr维度 -> 打乱训练集

```
import numpy as np
import copy
import time
import tensorflow as tf
import pickle
#生成
def batch_generator(arr, n_seqs, n_steps):
    #拷贝输入的数组，保证不改变原对象
    arr = copy.copy(arr)
    #单个训练集大小=单个数据大小*训练次数
    batch_size = n_seqs * n_steps
    #训练集个数为数据集长度除以batch集大小
    n_batches = int(len(arr) / batch_size)
    #将arr转换size
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    while True:
    #将数据集打乱
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            #生成一个和x相同的0矩阵
            y = np.zeros_like(x)
            #对时间序列
            #x=[a,b,c,d]则y=[b,c,d,a]
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y
```



`TextConverter`类初始化一个长度为50000的字典，并保存为pickle文件，如果文件已经存在直接打开文件，加载单词表vocab

如果不存在则按照如下步骤产生vocab，需要将单词中出现次数较少的舍弃：

set生成单词集合 -> 统计单词出现次数并存储在vocab_count中 -> 生成列表vocab_count_list元素为单词和出现次数 -> 将vocab_count_list按出现次数排列 -> 舍弃超过max_vocab的部分 -> 定义vocab

其中`word_to_int_table`为单词转换为数字的字典,`int_to_word_table`为数字转换为单词的字典.

```
class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                #使用pickle模块加载文件
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print(len(vocab))
            # 统计单词出现次数
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            #生成单词-频率列表
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            #当单词数超限值，提取限值内的部分
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            #生成前5000个单词的列表
            self.vocab = vocab
        #生成字典
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))
```

`vocab_size`统计单词数量

`word_to_int`将单词转换为对应数字

`int_to_word`将数字转换为对应单词，将不再字典中的单词输出为`<unk>`

`text_to_arr`使用`word_to_int`将文章所有的词转换为向量

`arr_to_text`使用`int_to_word`将数组转换为文章

`save_to_file`将单词保存

```
    #单词数量
    def vocab_size(self):
        return len(self.vocab) + 1
    #单词对应到数字
    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            #不在字典中的统一记作len(vocab)
            return len(self.vocab)
    #数字对应到单词
    #如果不在字典，输出<unk>
    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')
    #文章转换为数组
    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)
    #数组转换为文章
    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)
    #保存文件
    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)

```

**模型部分**

定义一个CharRNN类，初始化以下参数

        self.num_classes = num_classes #
        self.num_seqs = num_seqs #句子数量
        self.num_steps = num_steps #训练步
        self.lstm_size = lstm_size #lstm的层数
        self.num_layers = num_layers #神经网络层数
        self.learning_rate = learning_rate #学习率
        self.grad_clip = grad_clip #？？？？
        self.train_keep_prob = train_keep_prob #？？？
        self.use_embedding = use_embedding #？？
        self.embedding_size = embedding_size #？？
        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()
构造输入函数：

CNN神经网络是使用句子向量的前n-1个单词预测第n个单词，因此inputs和targets维度相同

定义输入序列数、样本值

`tf.one_hot(self.inputs, self.num_classes)`将inputs按照num_classes进行编码，张量中数据对应类为1其余参数为0

`tf.nn.embedding_lookup(embedding, self.inputs)`相当于numpy中按索引查找元素，其中inputs相当于索引

```
def build_inputs(self):
        with tf.name_scope('inputs'):
        #定义输入变量，有num_seqs个句子，num_steps列
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='inputs')
        #定义参数分类，和inputs维度相同        
            self.targets = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='targets')
            #定义droup_out的keep_prob参数
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            # 对于中文，需要使用embedding层
            # 英文字母没有必要用embedding层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

```

构建lstm神经网络







```
    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
        	#建立lstm实例
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            #设置每次有多少神经元不激活
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

```

定义损失函数

`tf.one_hot(self.targets, self.num_classes)`对targets进行one-hot编码

损失函数为交叉熵损失，关于交叉熵损失可以参考另外一篇文章

```
    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)
```



```
    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))
```



```
   def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

```





    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
    
        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)
    
        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
    
            c = pick_top_n(preds, vocab_size)
            samples.append(c)
    
        return np.array(samples)
    
    
    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
