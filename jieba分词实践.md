---
title: jieba分词实践
date: 2018-04-27 00:00:22
tags:
---

本文主要是jieba分词的学习记录。

<!-- more -->

```
import jieba
s = u'我想要在中广核的海滩边上走一走'

# cut方法

cut = jieba.cut(s)
list(cut)
['我', '想要', '在', '中广核', '的', '海滩', '边上', '走', '一', '走']
s = u'武汉市长江大桥和武汉市长姜大桥。'
```


全模式
尽量分成更多的词

```
','.join(jieba.cut(s,cut_all = True))
'武汉,武汉市,市长,长江,长江大桥,大桥,和,武汉,武汉市,市长,姜,大桥'
','.join(jieba.cut_for_search(s))
'武汉,武汉市,长江,大桥,长江大桥,和,武汉,武汉市,长姜,大桥'
```


获取词性可以用jieba.posseg

```
import jieba.posseg as psg
print([(x.word,x.flag) for x in psg.cut(s)])
[('武汉市', 'ns'), ('长江大桥', 'ns'), ('和', 'c'), ('武汉市', 'ns'), ('长', 'a'), ('姜', 'n'), ('大桥', 'ns')]
```


把姜前面的长识别成了形容词，哈哈哈
显示了每个词的词性
还可以对分词进行筛选，用startswith，获得名词

```
print([(x.word,x.flag) for x in psg.cut(s) if x.flag.startswith('n')])
[('武汉市', 'ns'), ('长江大桥', 'ns'), ('武汉市', 'ns'), ('姜', 'n'), ('大桥', 'ns')]
```


获取词频
Counter().most_common(20)
添加用户字典，定义用户字典
词语    词频 词性
姜大桥   5    'ns'
其中词频是一个数字，词性为自定义的词性，要注意的是词频数字和空格都要是半角的。
再进行分词

```
jieba.load_userdict('user_dict.txt')
print([(x.word,x.flag) for x in psg.cut(s)])
[('武汉市', 'ns'), ('长江大桥', 'ns'), ('和', 'c'), ('武汉市', 'ns'), ('长', 'a'), ('姜大桥', 'ns')]姜大桥已经变成人名了
```

