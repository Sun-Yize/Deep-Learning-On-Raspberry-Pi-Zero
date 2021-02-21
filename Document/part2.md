## 二、图像分类模型

### 2.1 模型选择

在得到花果数据集后，最重要的便是选择合适的**图像识别模**型进行学习，从而达到对水果与花卉分类的目的。

首先，我们对历年ILSVRC比赛中出现的经典模型进行学习，总结了多个网络的核心思想与优化方法，为后续模型的选择与优化奠定了基础。

由于花果数据集相对于ImageNet数据集较小，且需要的分类数目只有几十种，直接使用经典模型会使计算时间大大增加，并浪费一定的计算资源。并且最终的模型需要部署到**树莓派**中，所以网络的结构不能过大。

综合以上两点，我们选择了目前适用于移动设备的**轻型模型**，这类模型在保证一定的识别精度的情况下，大大减小了网络结构与参数量，从而减少网络规模。

在比较了常用的三种**轻型网络**（DenseNet、ShuffleNet、MobileNet）后，结合各模型特点与应用于花果数据集后得到的准确率，我们最终选择了**MobileNetV2**作为图像识别模型。

在沿用了原模型中深度可分离卷积与到残差结构的基础上，我们对模型网络结构与参数进行了调整与优化，使其更好地对数据集进行学习，并得到了比原模型更高的准确率。

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img207.png" width=90%>

#### 2.1.1 经典模型学习

在图像识别模型的选择过程中，我们首先学习了基于**ImageNet**数据集的ILSVRC比赛中的优秀经典模型，如Alexnet、VGG19、GoogleNet、ResNet等历年的冠/亚军模型。

**层数比较**：

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img201.png" width=50%>

**Top-1准确率**：

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img202.png" width=40%>

在这些网络中，我们感兴趣的模型有VGG、GoogleNet、ResNet，我们对这些模型的核心思想进行学习，分析了每个网络的结构以及优缺点，并找到各个网络相对于前一个网络的优化点，为我们之后自己的网络提供优化方向，总结如下：



**VGG**

采用连续的几个**3x3的卷积核**代替较大卷积核，作为网络中的卷积核大小搭建模型。

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img203.png" width=80%>

 

**优点**：结构简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2），通过不断加深网络结构可以提升性能

**缺点**：参数过多，耗费过多计算资源



**GoogleNet**

使用多个**inception模块**（如下图）串联从而形成最终的网络。

inception结构的主要贡献有两个：一是使用1x1的卷积来进行升降维，二是在多个尺寸上同时进行卷积再聚合。

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img204.png" width=50%>

**优点**：增加了网络的深度和宽度，但没有增加计算代价、提升了对网络内部计算资源的利用。



**ResNet**

参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了**残差块**（Residual Block）

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img205.png" width=80%>

**优点**：首次使用了残差块引入恒等快捷连接，直接跳过一个或多个层，解决了深度神经网络的“退化”问题，即给网络叠加更多的层后，性能却快速下降的情况。

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img206.png" width=40%>

#### 2.1.2 轻型网络的选择

通过将这些网络应用到我们的花果数据集，得出了较高的准确率，但综合考虑了各个网络参数量、计算时间以及花果分类的数据集后，我们决定选择轻型网络作为最终的模型，原因如下：

1. 由于该项目最终需要部署到**树莓派**上，经典网络模型的结构与参数量都过大，无法应用到树莓派
2. 经典网络模型的应用任务往往是大型分类任务，而本项目最终所需要分类的花果仅几十种，实际应用时导致计算资源与时间的浪费。
3. **轻型网络**：在保证一定的识别精度的情况下，大大减小网络结构与参数量，从而减少网络规模。适用于手机、树莓派等移动终端中，如DenseNet、ShuffleNet、MobileNet等。 



**DenseNet**

在ResNet的基础上提出了一个更激进的密集连接机制，互相连接所有的层。

网络由多个**dense block**组成，在dense block中每层的输入是前面所有层的输出concat形成的，结构如下： 

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img208.png" width=50%>

**优点**：加强了feature的传递，并更有效的利用了它、大大减少了参数的数量



**ShuffleNet**

在ResNeXt的基础上，使用**分组逐点卷积**（group pointwise convolution）来代替原来的结构。即通过将卷积运算的输入限制在每个组内，模型的计算量取得了显著的下降。

引入了**组间信息交换的机制**。即对于第二层卷积而言，每个卷积核需要同时接收各组的特征作为输入。 

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img209.png" width=80%>

**优点**：原创了三种混洗单元，每个单元都是由逐群卷积和信道混洗组成，这种结构极大的降低了计算代价



**MobileNetV1**

在VGG的基础上，将其中的标准卷积层替换为**深度可分离卷积**，其计算代价是由深度卷积和逐点卷积两部分。并添加了两个超参数：**瘦身乘子**（width multiplier）其取值范围为0~1，用来减少网络的通道数。另外一个参数为**分辨率乘子**（resolution multiplier），该参数将缩放输入图像的尺寸，尺寸范围为224~128之间。

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img210.png" width=80%>

**优点**：使用了深度可分离卷积，大大减少了参数量。并添加了两个超参数，使得在保证了一定准确度的前提下，网络模型进一步缩小



#### 2.1.3 MobileNetV2模型

在V2的网络设计中，除了继续使用V1中的深度可分离卷积之外，还使用了Expansion layer和 Projection layer。

**projection layer**使用1×1的网络结构，目的是将高维特征映射到低维空间去。

**Expansion layer**的功能正相反，在使用1×1的网络结构的同时，目的是将低维特征映射到高维空间。其中的一个超参数决定了将维度扩展几倍 

**网络结构**：先通过Expansion layer来扩展维度，之后用深度可分离卷积来提取特征，而后使用Projection  layer来压缩数据，让网络重新变小。因为Expansion layer 和 Projection  layer都具有可以学习的参数，所以整个网络结构可以学习到如何更好的扩展数据和重新压缩数据。

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img214.png" width=30%>

**优点**：在V1的基础上，使用了**倒残差结构**(Inverted residual block)，即使用Pointwise先对特征图进行升维，在升维后接上Relu，减少Relu对特征的破坏。并引入了**特征复用结构**（ResNet bottleneck） 

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img211.png" width=50%>

最终，通过将几种轻型网络应用于花果数据集后，综合**验证集准确率**以及**树莓派模型适用性**，我们选择了**MobileNetV2**为最终的网络模型，并进行了代码的调整与优化，使其更好地适用于本项目中的花果分类。

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img212.png" width=40%>


### 2.2 数据准备

我们需要使用我们已经预处理好的花卉与水果数据，图片共有30类，其中花卉与水果各有15类。

每类水果图片包含了1000张训练集，与100张测试集。数据集总共有33000张图片。

| 序号 |   英文名    | 中文名 |      | 序号 |    英文名     | 中文名 |
| :--: | :---------: | :----: | ---- | :--: | :-----------: | :----: |
|  1   |    apple    |  苹果  |      |  16  |     aster     |  紫苑  |
|  2   |   banana    |  香蕉  |      |  17  |    begonia    | 秋海棠 |
|  3   |  blueberry  |  蓝莓  |      |  18  |  calla lily   | 马蹄莲 |
|  4   |   cherry    |  樱桃  |      |  19  | chrysanthemum |  菊花  |
|  5   |   durian    |  榴莲  |      |  20  |  cornflower   | 矢车菊 |
|  6   |     fig     | 无花果 |      |  21  |   corydali    |  紫堇  |
|  7   |    grape    |  葡萄  |      |  22  |    dahlia     | 大丽花 |
|  8   |    lemon    |  柠檬  |      |  23  |     daisy     |  雏菊  |
|  9   |   litchi    |  荔枝  |      |  24  |    gentian    |  龙胆  |
|  10  |    mango    |  芒果  |      |  25  |  mistflower   |  雾花  |
|  11  |   orange    |  橙子  |      |  26  |    nigella    | 黑霉菌 |
|  12  |  pineapple  |  菠萝  |      |  27  |     rose      |  玫瑰  |
|  13  |    plum     |  李子  |      |  28  |   sandwort    |  沙参  |
|  14  | pomegranate |  石榴  |      |  29  |   sunflower   | 向日葵 |
|  15  | strawberry  |  草莓  |      |  30  |   veronica    | 婆婆纳 |

### 2.3 Tensorflow2框架搭建

我们使用Tensorflow2进行深度学习框架的搭建。理由如下：

* 我们在本项目中主要注重模型的最终使用。Tensorflow2在模型部署方面有着相当成熟的API，可以更加快速的进行部署。


* Tensorflow2有封装好的深度学习训练API，如tf.keras，能够快速的搭建模型和使用。

#### 2.3.1 预处理函数

首先我们导入需要使用的python包：

```python
import tensorflow as tf
from tensorflow.keras import layers, models
```

接下来我们定义图像处理函数，我们将图片所在文件夹名作为数据的标签，并将所有图片处理为相同的格式大小。

图片默认处理为224x224格式大小：

```python
def load_image(img_path,size = (224,224)):
    # 如果使用的windows系统，需要将sep='/'改为sep='\\'
    label = tf.cast(tf.compat.v1.string_to_number(tf.strings.split(img_path, sep='/',)[-2]), tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img,size)/255.0
    return(img,label)
```

（注意：load_image函数的label一行中，如果使用的为windows系统，需要将sep='/'改为sep='\\'，因为windows使用的文件名分隔符为\\）

定义模型训练过程中主要参数：

```python
BATCH_SIZE = 100
EPOCHS = 10
```

导入测试集与训练集，并行化处理数据：

```python
# 数据集文件夹所在路径
datapath = "/content/gdrive/MyDrive/data/classification/"

ds_train = tf.data.Dataset.list_files(datapath+"fruit_flower/train/*/*.jpg") \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
           .prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.Dataset.list_files(datapath+"fruit_flower/test/*/*.jpg") \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .batch(BATCH_SIZE) \
           .prefetch(tf.data.experimental.AUTOTUNE)
```

将预处理好的图片用matplotlib画图展示：

```python
from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
for i,(img,label) in enumerate(ds_train.unbatch().take(9)):
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img213.png" width=50%>

#### 2.3.2 定义MobileNetV2模型

使用Keras接口有以下3种方式构建模型：使用Sequential按层顺序构建模型，使用函数式API构建任意结构模型，继承Model基类构建自定义模型。

这里我们选择使用函数式API构建模型。

我们用Tensorflow2中的tf.keras.applications函数，调用MobileNetV2模型，其中：

+ 输入图片维度设置为(224,224,3)
+ 去除掉原有网络的卷积层
+ 定义新的卷积层

```python
# tf.keras.applications导入模型
Mo = tf.keras.applications.MobileNetV2(
    	input_shape=(224,224,3),
		include_top=False)
Mo.trainable=True

model = models.Sequential()
model.add(Mo)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(30, activation='sigmoid'))
model.summary()
```

```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_224 (Functi (None, 7, 7, 1280)        2257984   
_________________________________________________________________
global_average_pooling2d_2 ( (None, 1280)              0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 1280)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 512)               655872    
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_7 (Dense)              (None, 256)               131328    
_________________________________________________________________
dropout_5 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_8 (Dense)              (None, 30)                7710      
=================================================================
Total params: 3,052,894
Trainable params: 794,910
Non-trainable params: 2,257,984
_________________________________________________________________
```

#### 2.3.3 训练模型

接下来我们开始训练模型，首先设置模型的各个回调函数。其中分别包括：

+ TensorBoard：使用TensorBoard，将训练过程进行可视化。
+ ModelCheckpoint：设置模型的检查点，使模型下次训练时从检查点开始。
+ EarlyStopping：当训练损失函数连续多轮没有变化时，自动停止训练
+ ReduceLROnPlateau：根据训练的迭代次数，逐渐减小学习率，提高学习精度。

以上函数的具体代码如下：

```python
import datetime
import os

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callback_dir = datapath+'model/callback/'+stamp
tensorboard_callback = tf.keras.callbacks.TensorBoard(callback_dir, histogram_freq=1)

checkpoint_path = datapath+'model/checkpoint/'+stamp
model_save = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                verbose=1, 
                save_weights_only=True,
                period=20)

early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                              patience = 2, min_delta = 0.001, 
                              mode = 'min', verbose = 1)
```

导入最新的检查点：

```python
# 如果是初次训练，则不需要调用
checkpoint_dir = datapath+'model/checkpoint/'
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
```

定义模型的优化器以及损失函数：

+ 优化器：Adam优化器，即一种对随机目标函数执行一阶梯度优化的算法，该算法基于适应性低阶矩估计。
+ 损失函数：稀疏分类交叉熵（Sparse Categorical Crossentropy），多类的对数损失，适用于多类分类问题，且接受稀疏标签。

用Tensorflow2内置fit方法开始进行模型训练：

```python
model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"])

history = model.fit(ds_train,epochs=EPOCHS,validation_data=ds_test,
                    callbacks = [tensorboard_callback, model_save, early_stop, reduce_lr])
```

### 2.4 模型评估

#### 2.4.1 TensorBoard可视化

TensorBoard可以用于查看训练的进度如何，我们可以在命令行或jupyter notebook中调用TensorBoard，实时查看损失函数的下降情况以及训练的具体进度，这里我们用jupyter notebook进行查看，具体操作如下：

首先导入TensorBoard：

```python
%load_ext tensorboard

from tensorboard import notebook
notebook.list() 
```

接着输入训练对应文件夹，启用TensorBoard：

```python
notebook.start("--logdir "+datapath+"model/callback/")
```

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img216.png" width=100%>

#### 2.4.2 训练正确率查看

我们使用python的pandas包，将每次的迭代正确率和损失以表格的形式呈现：

```python
import pandas as pd

dfhistory = pd.DataFrame(history.history)
dfhistory.index = range(1,len(dfhistory) + 1)
dfhistory.index.name = 'epoch'

print(dfhistory)
```



除此之外，我们将训练中训练集和验证集的正确率与损失函数以图的方式呈现，具体如下：

```python
train_metrics = history.history["loss"]
val_metrics = history.history['val_loss']
epochs = range(1, len(train_metrics) + 1)
plt.plot(epochs, train_metrics, 'bo--')
plt.plot(epochs, val_metrics, 'ro-')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend(["train_loss", 'val_loss'])
plt.show()
```

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img217.png" width=60%>

```python
train_metrics = history.history["accuracy"]
val_metrics = history.history['val_accuracy']
epochs = range(1, len(train_metrics) + 1)
plt.plot(epochs, train_metrics, 'bo--')
plt.plot(epochs, val_metrics, 'ro-')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend(["train_accuracy", 'val_accuracy'])
plt.show()
```

<img src="C:/Users/86178/Desktop/深度学习/github/Document/image/img218.png" width=60%>

### 2.5 模型改进

在训练模型时要从模型的实际用途进行出发，我们进一步对训练好的模型进行改进。

在接下来的模型改进中，主要从以下两个方面进行考虑：

+ 减小模型的大小，提高模型运算速度
+ 提高模型的准确率

以上改进分别通过输入图片大小调整和网络参数调整来实现，同时我们还通过迁移学习提高模型训练效率。

#### 2.5.1 迁移学习

迁移学习通过下载以训练好的模型权重，减少模型的训练量，加快模型的训练速度。这里我们使用MobileNetV2在ImageNet上训练的权重作为迁移学习的模型。

首先定义模型，并调用迁移学习的参数：

```python
Mo = tf.keras.applications.MobileNetV2(
    	input_shape=(224,224,3),
	include_top=False,
	weights='imagenet')

model = models.Sequential()
model.add(Mo)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(30, activation='sigmoid'))
model.summary()
```

我们控制模型的前20层为不可训练的参数，20层往后为可训练参数：

```python
for layer in Mo.layers[:20]:
   layer.trainable=False
for layer in Mo.layers[20:]:
   layer.trainable=True
```

调用fit函数开始训练模型：

```python
history = model.fit(ds_train,epochs=EPOCHS,validation_data=ds_test,
                    callbacks = [tensorboard_callback, model_save, early_stop, reduce_lr])
```

#### 2.5.2 输入图片大小调整

输入图片的大小决定了卷积神经网络的复杂程度，以及参数的多少。对于较多的分类情况下，如ImageNet数据集，使用的是（224,224,3）维度的图片。但在本项目中，图片的分类较少，为30种，可以使用较小矩阵的图片，我们将图片大小调整为（100,100,3），并重新进行训练，具体如下。

改变函数，调整图片大小：

```python
def load_image(img_path,size = (224,224)):
    label = tf.cast(tf.compat.v1.string_to_number(tf.strings.split(img_path, sep='/',)[8]), tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img,size)/255.0
    return(img,label)
```

将图片进行可视化：

```python
plt.figure(figsize=(8,8))
for i,(img,label) in enumerate(ds_train.unbatch().take(9)):
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

重新对模型进行训练：

```python
history = model.fit(ds_train,epochs=EPOCHS,validation_data=ds_test,
                    callbacks = [tensorboard_callback, model_save, early_stop, reduce_lr])
```

#### 2.5.3 网络参数调整

卷积神经网络的参数，以及训练时的超参数很大程度决定了模型的好坏，模型的参数可以从以下几点进行调整：

+ 卷积神经网络参数
  + 网络层的激活函数，选择relu或sigmod
  + dropout层的添加以及参数改变，防止过拟合
  + 网络结构调整，删去冗余网络结构，减小计算量
+ 训练超参数选择
  + 优化器的选择，选择合适的优化器
  + 学习率大小的调整，通过减小学习率来提高精度

以下我们通过调整学习率，以及dropout层的参数，来对模型进行改进。

重新定义模型，调整模型参数：

```python
Mo = tf.keras.applications.MobileNetV2(
    	input_shape=(224,224,3),
		include_top=False)
Mo.trainable=True

model = models.Sequential()
model.add(Mo)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(30, activation='sigmoid'))
model.summary()
```

调整合适的学习率，将学习率从0.001调整至0.0005：

```python
model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0005),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"])
```

调用fit函数训练模型：

```python
history = model.fit(ds_train,epochs=EPOCHS,validation_data=ds_test,
                    callbacks = [tensorboard_callback, model_save, early_stop, reduce_lr])
```

