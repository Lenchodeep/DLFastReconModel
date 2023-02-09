## This is a project for deep learning based fast MR image reconstruction

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**1、Papers and links:**

DAGAN:  https://ieeexplore.ieee.org/document/8233175

Refine-GAN:  https://ieeexplore.ieee.org/abstract/document/8327637

---------------------------------------------------------------------------------------------------------------------------------------------------------



2、**深度学习服务器使用指南V1.0**

1> 远程连接：

1）当前服务器放置于实验室，可以通过**teamViewer**进行远程控制，**ip**地址10.0.0.18（需挂vpn）， 密码：lab7355608

![image-20210716010022151](assets/image-20210716010022151.png)

2）**ssh**连接，**visual Code** 

![image-20210716010059704](assets/image-20210716010059704.png)

2> 文件传输：

**WinScp**：

登陆配置：

![image-20210716010201278](assets/image-20210716010201278.png)

文件上传及下载：

公用文件夹：

请将代码、数据放于 **/home/d1/share/DLreconstruction/ **中自己建立的对应文件夹中。

![image-20210716010308361](assets/image-20210716010308361.png)



**3> 环境选择：**

1） python环境：****

当前服务器共配置三个python环境：**base**，**pytorchEnv**（pytorch环境），**tensorflowEnv**（tensorflow环境）：

<img src="assets/image-20210716010538203.png" alt="image-20210716010538203" style="zoom:150%;" />

运行代码之前需激活对应环境 **conda activate envName** 

1） IDE：当前只安装了visual Code作为IDE，可根据当习惯配置其他IDE（**spyder**， **pycharm**,  **jupyter**）

4> 其他问题

**1) GPU监测：**

Terminal中键入 **nvidia-smi**

<img src="assets/image-20210716010706248.png" alt="image-20210716010706248" style="zoom:150%;" />

**2) 关于tensor Flow环境下的vs code：**

Visual Code中的**code Runner**对t**ensor Flow**支持有Bug， 需要先在**terminal**中先激活对应的python环境，再在terminal中启动**Visual** **Code**（键入**code**）。

<img src="assets/image-20210716010843467.png" alt="image-20210716010843467" style="zoom:150%;" />

注意：在visual Code中需要选择对应的python环境

**3） 版本问题：**

当前安装的**pytorch**及**tensor Flow**均为最新版，由于用法等原因，**windows**下可以运行的代码可能会遇到问题，需要进行修改。

**4） Keras问题：**

**keras**目前只能使用**tensorflow**自带版本，不能使用额外安装的k**e**ras，目前还未解决。