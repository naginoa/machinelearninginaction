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

itchat.auto_login(True)
itchat.run(True)
```

### 命令行二维码

通过以下命令可以在登陆的时候使用命令行显示二维码：

```python

```

### 用户搜索


关于公众号、群聊的获取与搜索在文档中有更加详细的介绍。

### 附件的下载与发送

itchat的附件下载方法存储在msg的Text键中。

发送的文件的文件名（图片给出的默认文件名）都存储在msg的F
