# itchat

[![Gitter][gitter-picture]][gitter] ![py27][py27] ![py35][py35] [English version][english-version]

itchat是一个开源的微信个人号接口，使用python调用微信从未如此简单。

使用不到三十行的代码，你就可以完成一个能够处理所有信息的微信机器人。

当然，该api的使用远不止一个机器人，更多的功能等着你来发现，比如[这些][tutorial2]。

该接口与公众号接口[itchatmp][itchatmp]共享类似的操作方式，学习一次掌握两个工具。

如今微信已经成为了个人社交的很大一部分，希望这个项目能够帮助你扩展你的个人的微信号、方便自己的生活。

## 安装

可以通过本命令安装itchat：

```python
pip install itchat
```

## 简单入门实例

有了itchat，如果你想要给文件传输助手发一条信息，只需要这样：

```python
import itchat

itchat.auto_login()

itchat.send('Hello, filehelper', toUserName='filehelper')
```

如果你想要回复发给自己的文本消息，只需要这样：

```python
import itchat

@itchat.msg_register(itchat.content.TEXT)
def text_reply(msg):
    return msg.text

itchat.auto_login()
itchat.run()
```

一些进阶应用可以在下面的开源机器人的源码和进阶应用中看到，或者你也可以阅览[文档][document]。

## 试一试

这是一个基于这一项目的[开源小机器人][robot-source-code]，百闻不如一见，有兴趣可以尝试一下。

由于好友数量实在增长过快，自动通过好友验证的功能演示暂时关闭。

![QRCode][robot-qr]

## 截屏

![file-autoreply][robot-demo-file] ![login-page][robot-demo-login]

## 进阶应用

### 特殊的字典使用方式

通过打印itchat的用户以及注册消息的参数，可以发现这些值都是字典。

但实际上itchat精心构造了相应的消息、用户、群聊、公众号类。

其所有的键值都可以通过这一方式访问：

```python
@itchat.msg_register(TEXT)
def _(msg):
    # equals to print(msg['FromUserName'])
    print(msg.fromUserName)
```

属性名为键值首字母小写后的内容。

```python
author = itchat.search_friends(nickName='LittleCoder')[0]
author.send('greeting, littlecoder!')
```

### 各类型消息的注册

通过如下代码，微信已经可以就日常的各种信息进行获取与回复。

```python
import itchat, time
from itchat.content import *

@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING])
def text_reply(msg):
    msg.user.send('%s: %s' % (msg.type, msg.text))

@itchat.msg_register([PICTURE, RECORDING, ATTACHMENT, VIDEO])
def download_files(msg):
    msg.download(msg.fileName)
    typeSymbol = {
        PICTURE: 'img',
        VIDEO: 'vid', }.get(msg.type, 'fil')
    return '@%s@%s' % (typeSymbol, msg.fileName)

@itchat.msg_register(FRIENDS)
def add_friend(msg):
    msg.user.verify()
    msg.user.send('Nice to meet you!')

@itchat.msg_register(TEXT, isGroupChat=True)
def text_reply(msg):
    if msg.isAt:
        msg.user.send(u'@%s\u2005I received: %s' % (
            msg.actualNickName, msg.text))

itchat.auto_login(True)
itchat.run(True)
```

### 命令行二维码

通过以下命令可以在登陆的时候使用命令行显示二维码：

```python
itchat.auto_login(enableCmdQR=True)
```

部分系统可能字幅宽度有出入，可以通过将enableCmdQR赋值为特定的倍数进行调整：

```python
# 如部分的linux系统，块字符的宽度为一个字符（正常应为两字符），故赋值为2
itchat.auto_login(enableCmdQR=2)
```

默认控制台背景色为暗色（黑色），若背景色为浅色（白色），可以将enableCmdQR赋值为负值：

```python
itchat.auto_login(enableCmdQR=-1)
```

### 退出程序后暂存登陆状态

通过如下命令登陆，即使程序关闭，一定时间内重新开启也可以不用重新扫码。

```python
itchat.auto_login(hotReload=True)
```

### 用户搜索

使用`search_friends`方法可以搜索用户，有四种搜索方式：
1. 仅获取自己的用户信息
2. 获取特定`UserName`的用户信息
3. 获取备注、微信号、昵称中的任何一项等于`name`键值的用户
4. 获取备注、微信号、昵称分别等于相应键值的用户

其中三、四项可以一同使用，下面是示例程序：

```python
# 获取自己的用户信息，返回自己的属性字典
itchat.search_friends()
# 获取特定UserName的用户信息
itchat.search_friends(userName='@abcdefg1234567')
# 获取任何一项等于name键值的用户
itchat.search_friends(name='littlecodersh')
# 获取分别对应相应键值的用户
itchat.search_friends(wechatAccount='littlecodersh')
# 三、四项功能可以一同使用
itchat.search_friends(name='LittleCoder机器人', wechatAccount='littlecodersh')
```

关于公众号、群聊的获取与搜索在文档中有更加详细的介绍。

### 附件的下载与发送

itchat的附件下载方法存储在msg的Text键中。

发送的文件的文件名（图片给出的默认文件名）都存储在msg的F
