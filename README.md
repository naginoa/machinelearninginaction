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

## 简单`入门`实例

t))

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
