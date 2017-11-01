# 《机器学习实战》代码改进

《机器学习实战》是众所周知的机器学习的牛逼的教材

对此代码中的进行一些学习和改进


## 第二章

第一章是简介，所以直接从第二层开始

### 1.上下文管理器
```python
#讲文本记录转化为numpy
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        #去掉左右两边的空格，'\n'
        line = line.strip()
        #按照tab键进行分割
        listFromLine = line.split('\t')
        #将数据插入改列
        returnMat[index,:] = listFromLine[0:3]   
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
```


此处:

```python
fr = open(filename)
```
文件的使用没有使用上下文管理器，存在文件关闭隐患，详见项目`python-progressing`，应改为：
```python
with open(filename) as fr:
```

### 2.`python2` 和 `python3` 版本中`dict`的函数不同

书中是`python2`版本，多年之前比较流行，但是现在已经不在维护了，所以还是使用`python3`版本

```python
def classify0(inX, dataSet, labels, k):
    #dataset的长度
    dataSetSize = dataSet.shape[0]
    #tile()讲一个list,dict等转化为m*n的np向量
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    #列相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #从小到大排列，并提取索引值
    sortedDistIndicies = distances.argsort()
    #对label出现次数进行统计
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #如果没有则返回0
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #返回出现频率最高的label值和频率
    return sortedClassCount[0][0]
```

此处：

```python
sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
```

`python3`中字典已经没有.iteritems()函数，变为.items() ,因此会报错

改为：

```python
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
```

### 3.找出最小的错误率并制作错误率曲线图

此处书中没有。

测试集比率从0.1开始，如果测试集占比小于0.1，例如0.01，错误率会是0，但是此时测试集数据非常小，没有实际意义。因此从0.1开始

以0.02为梯度，到0.92.（同理，训练集比率过小也会导致没有意义等问题）。

代码：

代码修改为返回错误率

```python
#对约会网站进行测试
def datingClassTest(hoRatio):
    #0.1为测试集 0.9为训练集
    #hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        #print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    return (errorCount/float(numTestVecs))
    #print(errorCount)
```

进行循环开始计算不同占比的错误率

```python
hoRatio = 0.1
i = 0
hoRatios = np.zeros((41,2))
while(hoRatio <= 0.92):
    print('hoRatio is : ', hoRatio)
    a = kNN.datingClassTest(hoRatio)
    hoRatios[i] = np.tile([hoRatio, a],(1,1))
    hoRatio += 0.02
    i += 1
print(i)
```

讲列表进行操作修改为`numpy ndarray`

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100)
y1, y2 = hoRatios[:,0], hoRatios[:,1]
print(type(np.sin(x)))
plt.plot(y1, y2, label='error_rate')
plt.xlabel('hoRatio')
plt.ylabel('errorRate')
plt.legend()
plt.show()
```

结果图：

![Image text](https://github.com/naginoasukara/machinelearninginaction/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98/ch2/image/5.png)

### 用户搜索


关于公众号、群聊的获取与搜索在文档中有更加详细的介绍。

### 附件的下载与发送

itchat的附件下载方法存储在msg的Text键中。

发送的文件的文件名（图片给出的默认文件名）都存储在msg的F
