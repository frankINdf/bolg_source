### 微软小冰架构 ###

The Design and Implementation of XiaoIce,an Empathetic Social Chatbot
https://arxiv.org/pdf/1812.08989.pdf

主要内容：

## 小冰解决的问题 ##
EQ + IQ

技术：

CPS以多轮数为指标

基于MDPS 马尔科夫决策过程

将对话视为决策过程

__top_level process__：管理整体对话，选取技能和对话模式

__low-level process__：隶属于当前技能，是具体的动作行为，生成对话片段或者完成任务

__这样的设计有助于在技能间跳转，实现引导__

马尔科夫决策：属于强化学习，目的是得到最大CPS**应该很难收敛**

引导规则：平衡引导和利用信息


## 工程架构：##

__用户层__：前端, 提供交互层

__对话引擎__:

__核心闲聊(core chat)__：包括整体chat和领域chat

__技能__：小冰的内置技能，有task例如机票播放等，图片评论、内容创作

__共情计算__：用户理解、social skills(__这个是什么__)、小冰人设

__整体状态追踪__：追踪当前状态

__对话策略__: top level选取技能, top manager （__这个是什么__）


__数据层__：数据库，包括用户属性、小冰属性、主题索引、知识图谱、成对的数据


## DM详解：##


dialogue state $s$

dialogue policy $\pi$

action $\alpha = \pi(s)$

### global state tracker ###

追踪对话状态

对话状态是用户说话特征、小冰的回答、对话中的实体、共情特征

将working memory中的对话状态信息encde成\s


### dialogue policy ###

管理激活那些模块

__top_level policy__：二分类，核心闲聊或者技能; 并将query分发给low_level police

__这样做能解决什么问题__

输入是文本，激活Core Chat, Topic Manager 管理Core chat是否跳向新主题或者Domain Chat

用户输入图片或者视频，进入特定技能

tak compltion, deep engagement 和内容创作通过特定的用户输入进入

多个任务同时激活时

偏向留在一个技能，待完成之后再跳向其他技能


### Topic Manger ###

由一个分类器组成，决定是否状态跳转

用户感到无聊或者小冰知识不够时的策略（_怎么判断_）：

给一个兜底答复

重复用户输入

给其他候选答复

文本生成通过boosting tree（__怎么做__）来rank

_感觉有推荐的技术在里面_

boosting tree的特征：

1.上下文关系

2.主题需要更新（__根据什么判断__）

3.个人兴趣

4.大众化

5.用户接受率，根据历史的j


### Empathetic Computing ###

理解上下文 ==> encoding ==> 得到用户profile

计算EQ的模块，是核心？有哪些承上启下？

个人理解，主要是识别输入query的信息，存为结构化数据



重写用户输入$Q$和上下文$C$成$Q_c$

将$Q_c$和用户共情状态encode成$e_Q$

反馈$R$和反馈的共情向量$e_R$

输出的对话状态向量$s = (Q_c, C, e_Q, e_R)$

$s$会当做dailogue policy的一个特征，同时生成过程会用这些信息

__ Contextual Query Understanding(CQU) __ 信息来自：Named entity

identification、Co-reference resolution、sentence completion

_ User Understanding _ $e_Q$由key-value对组成, 信息来自:

_topic detection_

_Intent detection_

_sentiment analysis_

_Opinion detection_

__Interpersonal Response Generation__

对$e_Q$的响应

### Core Chat ###

出了生成对话，还有其他作用么

检索为主，其次是深度学习，无监督的检索是辅助

有基础的交流能力

__part1 General Chat__:开放域

__part2 Domain Chat__：专业域

将$e_R$信息编码到encoder的结果中

__Retrieval_Based Generator using Paired Data__

400response作为候选

情感检索依赖paired dataset

结果生成的方法：
__Neural Response Generator__

neural-model-based generator 提供高覆盖

retrieval-based 提供高质量的内容

beam search 的到20个候选

__Retrieval-Based Generator using Unpaired Data__

non-conversational data用来提升覆盖和质量

使用KG来扩充query,格式$(h, r, t)$即（head, relation, tail），

只有当head在问题中，tail在回答中才会用

__Response Candidate Ranker__

用boost tree来rank

rank时用的特征：

__Local cohesion feature__ 用DSSMs

__Global coherence feature__ 比较语义的一致性，用DSSMs

__Empathy matching feature__ 比较情感的特征

__Retrieval matching feature__ 比较相似度，用BM25和TFIDF

得分的含义：

0：可能终止对话

1：可以接受

2：能够共情

__兜底答复__

使用偏向话题继续的兜底
