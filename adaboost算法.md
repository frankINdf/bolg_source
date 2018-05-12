---
title: adaboost算法
date: 2018-03-29 22:28:37
tags: 机器学习
---



​        adaboost算法的核心思想就是由分类效果较差的弱分类器逐步的强化成一个分类效果较好的强分类器。而强化的过程，就是逐步的改变样本权重，样本权重的高低，代表其在分类器训练过程中的重要程度。

<!-- more -->

该算法首先定义辅助函数stumpClassify，输入数据，分界值后输出分类的结果

```
def stumpClassify(dataIn,dimen,threshVal,threshIneq):
    #输入数据、采用哪列特征分类、分界的限值，是大于还是小于
    m=dataIn.shape[0]
    retArray=np.ones((m,1))
    if threshIneq=='lt':
    #该分类是小于，则大于分界值得数据预测错误，定义为-1
        retArray[dataIn[:,dimen]<=threshVal]=-1.0
    else:
    #该分类是大于，则小于分界值得数据预测错误
        retArray[dataIn[:,dimen]>threshVal]=-1.0
    return retArray
```

函数使用方法：

```
#定义输入数据
dataIn=np.arange((12)).reshape(6,2)
Out[43]: 
array([[ 0,  1],
       [ 2,  3],
       [ 4,  5],
       [ 6,  7],
       [ 8,  9],
       [10, 11]])
#以小于2的值作为第2列的分界点
stumpClassify(dataIn,1,2,'lt')
Out[47]: 
array([[-1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [ 1.]])
```

​          有了分类函数，接着需要定义buildStump选取当前情况下，最佳的分类点

```
def buildStump(dataIn,classLabel,D):
#D为输入的分类权重，后面会用到
    #输入数据，正确分类，迭代系数
    n=dataIn.shape[1]
    m=dataIn.shape[0]
    numStep=10.0
    bestStump={}
    bestClass=np.zeros((m,1))
    minError=np.inf
    for i in range(0,n):
        iAxis=dataIn[:,i]
        minAxis=min(iAxis)
        maxAxis=max(iAxis)
        stepSize=(maxAxis-minAxis)/numStep
    	#考虑大于和小于两种分类情况
        for ineq in ['lt','gt']:
            for j in range(-1,int(numStep)+1):
            #每次计算错误值
                threshVal=minAxis+stepSize*float(j)
                #从threshVal值处对i特征值对应数据进行分类
                predictCategory=stumpClassify(dataIn,i,threshVal,ineq)
                #初始化误差值矩阵
                errArr=np.ones((m,1))
                #预测正确的点误差矩阵值为0，其他点为1
                errArr[predictCategory==classLabel]=0
                #误差矩阵乘以权重得到该threshVal的分类总误差，
                errSum=np.dot(D.T,errArr)
                #更新误差最小的threshVal
                if errSum<minError:
                    minError=errSum
                    bestClass=predictCategory.copy()
                    bestStump['dim']=i
                    bestStump['threshVal']=threshVal
                    bestStump['ineq']=ineq
    print('j',j,'ineq',ineq,'split',i,'\nthresh',threshVal,'\nerrorsum',errSum)
    return bestStump,minError,bestClass
    
   classLabel=np.array(([[1],[-1],[-1],[1],[1],[1]]))    
   buildStump(dataIn,classLabel,D)
Out[56]: 
#得到当前权重下最佳分类
#分类的特征为0列对应的特征，分类值为4，错误率0.167，分类结果[-1,-1,-1,1,1,1]
({'dim': 0, 'ineq': 'lt', 'threshVal': 4.0},
 array([[ 0.16666667]]),
 array([[-1.],
        [-1.],
        [-1.],
        [ 1.],
        [ 1.],
        [ 1.]]))
```

​        得到了一列一种情况下的分类方法，就相当于有了一个分类器，adaboost的核心是将多个分类器的结果按照权重相加，得到最后的结果。下面通过addBoostTrainDS来训练得到各个分类器的权重。

```
def addBoostTrainDS(dataIn,classLabels,numIt=40):
    weakClassArr=[]
    m=dataIn.shape[0]
	#初始化D为1/m，m为数据个数
    D=np.ones((m,1))/m
    aggBestClass=np.zeros((m,1))
    for i in range(numIt):
     	#得到每次更新权重值后的最佳分类
        bestStump,error,bestClass=buildStump(dataIn,classLabel,D)
        #求出误差率alpha
        alpha=0.5*np.log((1-error)/max(error,1e-12))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        #每一项的权重
        expon=-1*alpha*(classLabels.T*bestClass)
        print('expon',expon)
        D=D*np.exp(expon)
        D=D/D.sum()#更新D值
        aggBestClass+=alpha*bestClass
        #类别为1和-1
        aggErrors=np.ones((m,1))
        samePred=np.where(np.sign(aggBestClass)==classLabel)
        difPred=np.where(np.sign(aggBestClass)!=classLabel)
        aggErrors[samePred]=0
        aggErrors[difPred]=1
        print('D',D,'best',bestClass,'\naggBestClass',aggBestClass  
        errorRate=aggErrors.sum()/m
        print('totalError',errorRate)
        if errorRate==0: break
    return weakClassArr,aggErrors
```

运行结果

```

```

