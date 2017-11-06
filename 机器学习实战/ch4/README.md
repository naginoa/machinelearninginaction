## 第四章

以下是网上寻找加整理的一些资料书中没有

朴素贝叶斯分类器

朴素贝叶斯分类是一种十分简单的分类算法，叫它朴素贝叶斯分类是因为这种方法的思想真的很朴素，朴素贝叶斯的思想基础是这样的：对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。通俗来说，就好比这么个道理，你在街上看到一个黑人，我问你你猜这哥们哪里来的，你十有八九猜非洲。为什么呢？因为黑人中非洲人的比率最高，当然人家也可能是美洲人或亚洲人，但在没有其它可用信息下，我们会选择条件概率最大的类别，这就是朴素贝叶斯的思想基础。      

朴素贝叶斯分类的正式定义如下：      

1、设<img src="http://latex.codecogs.com/gif.latex?x=\{a_1,a_2,...,a_m\}">为一个待分类项，而每个a为x的一个特征属性。

2、有类别集合<img src="http://latex.codecogs.com/gif.latex?C=\{y_1,y_2,...,y_n\}">。

3、计算<img src="http://latex.codecogs.com/gif.latex?P(y_1|x),P(y_2|x),...,P(y_n|x)">。

4、如果<img src="http://latex.codecogs.com/gif.latex?P(y_k|x)=max\{P(y_1|x),P(y_2|x),...,P(y_n|x)\}">，则<img src="http://latex.codecogs.com/gif.latex?x%20\in%20y_k">。      

那么现在的关键就是如何计算第3步中的各个条件概率。我们可以这么做：      

1、找到一个已知分类的待分类项集合，这个集合叫做训练样本集。      

2、统计得到在各类别下各个特征属性的条件概率估计。即<img src="http://latex.codecogs.com/gif.latex?P(a_1|y_1),P(a_2|y_1),...,P(a_m|y_1);P(a_1|y_2),P(a_2|y_2),...,P(a_m|y_2);...;P(a_1|y_n),P(a_2|y_n),...,P(a_m|y_n)">。      

3、如果各个特征属性是条件独立的，则根据贝叶斯定理有如下推导：<img src="http://latex.codecogs.com/gif.latex?P(y_i|x)=\frac{P(x|y_i)P(y_i)}{P(x)}">

因为分母对于所有类别为常数，因为我们只要将分子最大化皆可。又因为各特征属性是条件独立的，所以有：
<img src="http://latex.codecogs.com/gif.latex?P(x|y_i)P(y_i)=P(a_1|y_i)P(a_2|y_i)...P(a_m|y_i)P(y_i)=P(y_i)\prod^m_{j=1}P(a_j|y_i)">

根据上述分析，朴素贝叶斯分类的流程可以由下图表示（暂时不考虑验证）：

<img src="http://images.cnblogs.com/cnblogs_com/leoo2sk/WindowsLiveWriter/4f6168bb064a_9C14/1_2.png">

可以看到，整个朴素贝叶斯分类分为三个阶段：

第一阶段——准备工作阶段，这个阶段的任务是为朴素贝叶斯分类做必要的准备，主要工作是根据具体情况确定特征属性，并对每个特征属性进行适当划分，然后由人工对一部分待分类项进行分类，形成训练样本集合。这一阶段的输入是所有待分类数据，输出是特征属性和训练样本。这一阶段是整个朴素贝叶斯分类中唯一需要人工完成的阶段，其质量对整个过程将有重要影响，分类器的质量很大程度上由特征属性、特征属性划分及训练样本质量决定。

第二阶段——分类器训练阶段，这个阶段的任务就是生成分类器，主要工作是计算每个类别在训练样本中的出现频率及每个特征属性划分对每个类别的条件概率估计，并将结果记录。其输入是特征属性和训练样本，输出是分类器。这一阶段是机械性阶段，根据前面讨论的公式可以由程序自动计算完成。

第三阶段——应用阶段。这个阶段的任务是使用分类器对待分类项进行分类，其输入是分类器和待分类项，输出是待分类项与类别的映射关系。这一阶段也是机械性阶段，由程序完成。

### 疑问 

对书中第四章有几处疑问：

#### 1. 认为此处代码可以修改
    
```python
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)
```

我认为此处代码是一句话中的每个单词对dataset对空集做并集操作，从此达到去重的效果。那何必不妨改成一下代码：

```python
def createVocabList2(dataSet):
    vocabSet = set(dataSet)  #create empty set
    return list(vocabSet)
```

结果会出现如下错误

`TypeError: unhashable type: 'list'`

原因是dataset传入的参数并不是一句话，而是几句话，代码是对每句话取并集。

#### 2.对<img src="http://latex.codecogs.com/gif.latex?P(w|c_i)=P(w_0,w_1,w_2...w_n|c_i)">的代码实现有所不理解

```python
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords); p1Num = zeros(numWords)      #change to ones() 
    p0Denom = 0.0; p1Denom = 0.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = (p1Num/p1Denom)          #change to log()
    p0Vect = (p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive
```

此处不理解为什么 p1Num/p1Denom 就是这个概率，并且为什么没有除以C？

1.数学上有 几率 和 概率 两种概念。以抛硬币为例，0.5为硬币正面向上的几率，而硬币向上的概率是实验次数中向上的次数除以总实验次数。例如48/100.

2.结合本实验训练集有六句话，总共的词汇表的向量应该有20-30的长度。每一个单词若出现则在他的索引处加一，最后除以该句话的单词数总量。其中如下代码就是这样的思想

```python
p1Num += trainMatrix[i]
p1Denom += sum(trainMatrix[i])
```

计算结果：

```python
[ 0.125       0.          0.04166667  0.04166667  0.04166667  0.
  0.04166667  0.04166667  0.04166667  0.04166667  0.04166667  0.04166667
  0.04166667  0.04166667  0.04166667  0.          0.04166667  0.04166667
  0.04166667  0.          0.04166667  0.          0.04166667  0.08333333
  0.          0.          0.04166667  0.          0.04166667  0.          0.
  0.        ]
  ```

### 纠错及改进

#### 1.上下文管理器
详情参照 ------->  <a href='https://github.com/naginoasukara/python-progress/blob/master/context/Context_manager.py' value='上下文管理器详情'>上下文管理器详情</a>

bayes.py的spamTest()函数中出现错误

```python
   for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        #append 是将 变量原本的方式加进去，比如说把list填入。extend是把元素填入，把list中的元素填入
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
```

出现

```python
'gbk' codec can't decode byte 0xae in position 199: illegal multibyte sequence
```

原因只源代码中直接使用open()打开文件，这样在工程中是非常不科学的，尤其当数据量大时会引发各种错误，应改为：

```python
    for i in range(1,26):
        with open('email/spam/%s.txt' % i ,) as f:
            wordList = textParse(f.read())
        #append 是将 变量原本的方式加进去，比如说把list填入。extend是把元素填入，把list中的元素填入
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        #上下文管理器
        with open('email/ham/%s.txt' % i ,) as f:
            wordList = textParse(f.read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
```

#### 2.bayes.py的spamTest()函数中出现错误

```python
TypeError: 'range' object doesn't support item deletion
```

```python
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        #随机产生一个>=0并且<50的整数
        randIndex = int(random.uniform(0,len(trainingSet)))
        #将它加入测试集
        testSet.append(trainingSet[randIndex])
        #并从训练集中删除
        del(trainingSet[randIndex])  
    #存放向量和标签值
    trainMat=[]; trainClasses = []
```

python2中的range返回的是一个列表
python3中的range返回的是一个迭代值

因此将range(50)改为列表生成式

```python
    trainingSet = [i for i in range(50)]; testSet=[]           #create test set
    for i in range(10):
        #随机产生一个>=0并且<50的整数
        randIndex = int(random.uniform(0,len(trainingSet)))
        #将它加入测试集
        testSet.append(trainingSet[randIndex])
        #并从训练集中删除
        del(trainingSet[randIndex])  
    #存放向量和标签值
    trainMat=[]; trainClasses = []
```

#### 3.交叉验证是通过多次不同的训练集测试集划分，来做错误率最低的划分方式，甚至是修正分类器，此处没有求平均和修正。那么我先做求平均和最低，修正分类器在第七章会讲到

```python
import numpy as np

k = 1
i = 0
errors = []
xylabel = np.zeros((10,2))
while(k <= 10):
    a = bayes.spamTest()
    errors.append(a)
    xylabel[i] = np.tile([k, a],(1,1))
    print(xylabel[i])
    i += 1
    k += 1
print(k)
```

```python
print('average errorRate is :', sum(errors)/len(errors))
average errorRate is : 0.08
```

```python
import matplotlib.pyplot as plt

y1, y2 = xylabel[:,0], xylabel[:,1]
plt.plot(y1, y2, label='error_rate')
plt.xlabel('k')
plt.ylabel('errorRate')
plt.legend()
plt.show()
```

如下图：

![Image text](https://github.com/naginoasukara/machinelearninginaction/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98/ch4/image/1.png)

可见，在训练集测试集数据量不是十分大时，会出现0错误的情况，并且训练速率要比KNN快.















