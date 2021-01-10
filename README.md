# 树莓派zero图像分类与目标检测

> 山东大学（威海） 2018级数据科学与人工智能实验班
>
> 孙易泽 吴锦程 詹沛 徐潇涵

**树莓派zero图像分类与目标检测**是深度学习的研究项目，旨在通过深度学习算法，实现树莓派的**实时识别与分类**。

在树莓派上运行深度学习主要分为以下几个步骤：

+ 首先是**数据的获取及预处理**，图像分类与目标检测需要大量**干净且高质量**的图片数据进行训练，我们需要通过不同方式，尽可能多的获取到相关的图片数据，并处理为深度学习可用的形式。
+ 接下来先实现**图像分类**，根据深度学习领域存在的相关模型，选择适合于树莓派上运行的深度学习模型。通过**Tensorflow2**搭建深度学习框架，通过对模型参数不断调整，训练出正确率高且能快速运行的模型。通过对模型的不断改进，在保持模型正确率的同时，减小模型的大小。
+ **目标检测模型**也是一个侧重点，我们选择轻量级的深度学习模型，并使用**Tensorflow2 Object Detection**进行模型的训练，能够进行水果和花卉物体的准确检测，做到一张图片中正确识别多个不同物体的位置与种类。
+ 最后是**图像分类模型与目标检测模型分别的部署**，将训练好的模型部署到树莓派中，并利用摄像头实时对数据进行处理，做到图片的实时检测。

以下是详细的说明文档：

+ 1.[树莓派zero图像分类与目标检测—数据获取与预处理](https://github.com/Sun-Yize-SDUWH/Deep-Learning-On-Raspberry-Pi-Zero/blob/master/Document/part1.md)

+ 2.[树莓派zero图像分类与目标检测—图像分类模型](https://github.com/Sun-Yize-SDUWH/Deep-Learning-On-Raspberry-Pi-Zero/blob/master/Document/part1.md)

+ 3.[树莓派zero图像分类与目标检测—目标检测模型](https://github.com/Sun-Yize-SDUWH/Deep-Learning-On-Raspberry-Pi-Zero/blob/master/Document/part1.md)

+ 4.[树莓派zero图像分类与目标检测—深度学习部署](https://github.com/Sun-Yize-SDUWH/Deep-Learning-On-Raspberry-Pi-Zero/blob/master/Document/part1.md)

除此之外，本项目还在其他平台有相关的开源资料：

[知乎专栏—树莓派zero图像分类与目标检测](https://www.zhihu.com/column/c_1326223429637902336)

[b站视频链接](https://www.bilibili.com/video/BV12a4y1n7MW)

