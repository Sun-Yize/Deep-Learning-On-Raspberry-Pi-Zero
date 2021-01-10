## 四、深度学习部署

### 4.1 单片机简介

#### 4.1.1 硬件简介

<img src="image/img401.png" width=50%>

##### 树莓派zero w

在用树莓派部署深度学习过程中，我们选用树莓派zero w作为主要设备，树莓派zero w是树莓派系列最为基础的设备，搭载树莓派linux系统下，可以很好的运行程序。同时它还包括了wifi模块与蓝牙模块，方便pc与树莓派之间数据的传输。

树莓派zero主要参数如下：

- 博通 BCM2835 芯片 1GHz ARM11 core
- 512MB LPDDR2 SDRAM
- 一个 micro-SD 卡槽
- 一个 mini-HDMI 接口，支持 1080p 60hz 视频输出
- Micro-USB 接口用于供电和数据传输
- MicroUSB数据线，8G内存的MicroSD卡，用于烧制linux系统镜像

树莓派zero相比于其他型号树莓派，性能略有差异，但是仍可以胜任模型部署。

##### 摄像头

因为在训练模型过程中，我们对图片没有过高要求，仅采用较小像素图片进行训练，所以在实际使用时，我们使用500万像素摄像头进行拍摄，在实际使用中能够充分的发挥其作用。

##### 3.5寸显示屏

显示屏采用串口外接3.5寸显示屏，主要用于展示图像分类与目标检测的具体结果，屏幕为LCD显示屏，具有触摸功能，可以对系统进行具体的操控。

#### 4.1.2 软件环境

我们使用了**Raspberry Pi OS + python3.7**作为我们的软件环境。

Raspberry Pi OS环境自带python2.X版本，但是我们深度学习框架需要3.X以上的版本，所以需要在Linux系统中配置python环境。

在python官网下载后，选择源码安装，在通过xshell拷贝到linux系统中。通过文件传输将下载的压缩包上传后，通过yum-y命令安装依赖包和tar命令解压源码包。

```
./configure --prefix=/home/python3   
```

使用该命令为将要添加的python安装环境变量，在建立一个sh文件添加环境变量进去之后重载一下，linux系统下的python环境就配置完成了

### 4.2 树莓派环境搭建

#### 4.2.1 Raspberry Pi OS系统配置

##### 1.系统下载

我们使用Raspberry Pi Imager在SD卡上进行快速安装，首先在树莓派官网下载Raspberry Pi Imager：

[Raspberry Pi Imager下载](https://www.raspberrypi.org/software/)

<img src="image/img402.png" width=100%>

下载完成后，我们打开安装器，选择Raspberry Pi OS系统，并选择对应的SD卡进行系统安装。

<img src="image/img403.png" width=40%>

等待下载结束后，我们便得到了一张装有树莓派系统的SD卡。

##### 2.文件配置

我们将SD卡插入树莓派，并按照系统提示完成系统的安装：

<img src="image/img404.png" width=60%>

接下来我们还需要对系统进行简单的配置。

+ root账户设置

  首先设置root账户密码：

  ```
  sudo passwd root
  ```

  接下来我们编辑文件，配置root远程登录的权限：

  ```
  nano /etc/ssh/sshd_config
  ```

  打开文件后，在文档末尾添加：

  ```
  PermitRootLogin yes
  PermitEmptyPasswords no
  PasswordAuthentication yes
  ```

  添加完成后，用ctrl+o 保存，ctrl+x 退出。

+ 摄像头连接树莓派

  首先将摄像头与树莓派相连，接着在命令行中输入：

  ```
  sudo raspi-config
  ```

  选择Interface Options—camera，选择yes，将摄像头权限开启，我们便可以使用树莓派进行摄像头拍照了。

  在命令行执行如下命令：

  ```
  raspistill -t 2000 -o image.jpg
  ```

  如果看到文件夹中新增了image.jpg文件，则代表配置成功。

#### 4.2.2 Tensorflow2安装

tensorflow lite支持树莓派3及以上的版本，如果使用的是以上版本的树莓派，则可以到以下网址进行tensorflow lite的下载和安装。

[tensorflow lite下载](https://www.tensorflow.org/lite/guide/python?hl=zh-cn)

由于树莓派zero不支持tensorflow lite，我们必须下载完整的Tensorflow2包，再从中调用Tensorflow lite模块。

以下是树莓派zero安装tensorflow2的具体方法。首先我们需要下载tensorflow2的arm编译版本，在[tensorflow arm编译版本下载](https://github.com/lhelontra/tensorflow-on-arm/releases/tag/v2.2.0)可以找到对应支持的版本。

因为我们使用的是python3.7，所以我们在树莓派命令行中输入：

```
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.2.0/tensorflow-2.2.0-cp37-none-linux_armv6l.whl
```

下载完成后对文件进行重命名：

```
mv tensorflow-2.2.0-cp37-none-linux_armv6l.whl tensorflow-2.2.0-cp37-abi3-linux_armv6l.whl
```

然后使用pip3安装对应的.whl文件

```
sudo pip3 install tensorflow-2.2.0-cp37-abi3-linux_armv6l.whl
```

等待程序安装好后，我们便可以在树莓派zero上使用Tensorflow2了。输入如下命令进行测试：

```
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([100, 100])))"
```

如果出现了正确的输出，则代表tensorflow2安装成功。

#### 4.2.3 OpenCV安装

OpenCV的全称是Open Source Computer Vision Library，是一个跨平台的计算机视觉库，我们利用OpenCV操作树莓派进行拍照和图像的预处理。OpenCV在树莓派zero上的安装方法具体如下。

首先在命令行输入以下内容，安装必要的环境配置：

```
sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev
```

接下来我们使用pip3安装OpenCV：

```
pip3 install opencv-python==3.4.6.27
```

等待安装成功后，我们便可以使用OpenCV了。

### 4.3 树莓派部署模型

#### 4.3.1 图像分类模型部署

##### 1.导出为tensorflow模型

模型训练好之后会通过lastest_checkpoint命令导入最后一次训练的参数，checkpoint_dir是运行过程中得到的网络结构和权重值，作为暂时的值存储在文件夹里

```python
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
```

模型结构参数导出后，需要在重新运行一次，运行结果应该与训练过程的最后一次结果相同。

```python
model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )

history = model.fit(ds_train,epochs=500,validation_data=ds_test,
                    callbacks = [tensorboard_callback, cp_callback])

```

此时的model包括了网络结构和权重参数，可以直接保存为h5文件，这里得到的h5文件大小为28.7M

```python
model.save('./data/moblie_2.h5')
```

##### 2.使用tflite部署

tflite是谷歌自己的一个轻量级推理库。主要用于移动端。之前的tensorflow mobile就是用的tflite部署方式，tflite使用的思路主要是从预训练的模型转换为tflite模型文件，拿到移动端部署。tflite的源模型可以来自tensorflow的saved model或者frozen model,也可以来自keras。

```python
model=tf.keras.models.load_model("./data/moblie_2.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open ("model.tflite" , "wb") .write(tfmodel)
```

通过此代码读取保存的h5文件经过convert处理后转换成tflite文件，此时得到的文件大小只有7.4M,大大的减小了模型大小。

##### 3.摄像头拍照

通过opencv包打开摄像头进行拍摄

```python
import cv2 as cv
def video_demo():
#0是代表摄像头编号，只有一个的话默认为0
    capture=cv.VideoCapture(0) 
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while(True):
        ref,frame=capture.read()
 
        cv.imshow("1",frame)
#等待30ms显示图像，若过程中按“Esc”退出
        c= cv.waitKey(30) & 0xff 
        if c==27:
            capture.release()
            break    
```

cv.VideoCapture(0)表示读取视频，当输入为0时默认打开的是电脑摄像头。
read函数会返回两个值ref和frame，前者为true的时候表示获取到了图像，后者参数表示截取到的每一张图片。
cv.waitKey(30)&oxff: cv.waitKey(delay)函数如果delay为0就没有返回值，如果delay大于0，如果有按键就返回按键值，如果没有按键就在delay秒后返回-1，0xff的ASCII码为1111 1111，任何数与它&操作都等于它本身。`Esc`按键的ASCII码为27，所以当c==27时，摄像头释放。

```python
video_demo()
cv.destroyAllWindows()
```

最后通过cv.destroyAllWindows()函数清除所有的方框界面。

#### 4.3.2 目标检测模型部署

##### 1.导入tflite模型

首先我们需要在树莓派上下载Tensorflow Object Detection的API包，在树莓派命令行中输入：

```
git clone https://github.com/tensorflow/models
```

克隆完成后，将克隆的仓库进行重命名：

```
mv models-master models
```

下载目标检测API必要的软件包：

```
pip3 install tf_slim
pip3 install lvis
```

导入python的环境路径：

```
export PYTHONPATH=$PYTHONPATH:models/research/:models
```

接下来我们便可以进行目标检测模型的部署了。部署主要分为两部分，首先是加载tflite模型：

```python
import tensorflow as tf
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder
import cv2


# 模型识别种类个数
num_classes = 16
# 模型位置
pipeline_config = 'pipeline.config'
# 模型标签
category_index = {1: {'id': 1, 'name': 'apple'}, 2: {'id': 2, 'name': 'banana'}, 3: {'id': 3, 'name': 'grape'}, 4: {'id': 4, 'name': 'kiwifruit'}, 5: {'id': 5, 'name': 'mango'}, 6: {'id': 6, 'name': 'orange'}, 7: {'id': 7, 'name': 'pear'}, 8: {'id': 8, 'name': 'stawberry'}, 9: {'id': 9, 'name': 'calla lily'}, 10: {'id': 10, 'name': 'cornflower'}, 11: {'id':11, 'name': 'corydalis'}, 12: {'id': 12, 'name': 'dahlia'}, 13: {'id': 13, 'name': 'daisy'}, 14: {'id': 14, 'name': 'gentian'}, 15: {'id': 15, 'name': 'nigella'}, 16: {'id': 16, 'name': 'sunflower'}}

# 定义模型
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
model_config.ssd.num_classes = num_classes
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(model_config=model_config, is_training=True)

# 加载tflite文件
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
label_id_offset = 1
```

##### 2.OpenCV拍照与展示

接下来用OpenCV包进行实时拍照处理，并将拍照结果放入目标检测模型进行检测：

```python
# 定义摄像头
capture = cv2.VideoCapture(0)

while True:
    # 拍照并预处理照片
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test = np.expand_dims(frame_rgb, axis=0)
    input_tensor = tf.convert_to_tensor(test, dtype=tf.float32)
    # 目标检测模型进行检测
    boxes, classes, scores = detect(interpreter, input_tensor)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        test[0],
        boxes[0],
        classes[0].astype(np.uint32) + label_id_offset,
        scores[0],
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    # 呈现检测结果
    frame = cv2.cvtColor(test[0], cv2.COLOR_BGR2RGB)
    cv2.imshow("Object detector", frame)
    c = cv2.waitKey(20)
    # 如果按q键，则终止
    if c == 113:
        break
cv2.destroyAllWindows()
```

### 4.4 服务器改进部署方式

#### 4.4.1 Flask框架的搭建

Flask是一个使用python编写的轻量级web应用框架，主要用来接收和发送数据。当树莓派端Flask发送Post请求时，Flask可以使用Request包获取传来的数据，并将计算结果作为Post请求的返回值返回给树莓派

在服务器中使用Flask框架时，我们需要引入flask包，并定义函数，这样当接收到树莓派请求时，程序便会执行对应的函数，并将结果返回给树莓派。

以下是一个简单的Flask框架搭建：

```python
from flask import Flask
app = Flask(__name__)

@app.route('/', methods=["post"])
def index():
    return "<h1 style='color:red'>hello world!</h1>"

if __name__ == '__main__':
    app.run(host='192.168.1.1', debug=True)
```

云服务器使用flask时，只需要将端口号对应的函数上面使用装饰器，并在主函数运行主端口号即可。

```python
if __name__ == '__main__':
    app.run(host='192.168.1.1', debug=True, port=8888)
```

#### 4.4.2 Nginx+uwsgi的配置

单纯使用flask框架构造简单,但是器并发性效果较差，我们可以改进部署方式，选用Nginx + uwsgi + flask的部署方式增加稳定性。

首先安装Nginx，用如下命令进行安装：

```
apt install nginx
```

安装完成后对Nginx进行配置，具体的配置因服务器的具体情况而定：

```
server {
        listen 80;  # 监听端口，http默认80
        server_name _; # 填写域名或者公网IP
        location / {
                include uwsgi_params;   # 使用nginx内置的uwsgi配置参数文件
                uwsgi_pass 127.0.0.1:8088;   # 转发请求到该地址端口
                uwsgi_param UWSGI_SCRIPT main:app;   # 调用的脚本名称和application变量名
        }
}
```

最后启动Nginx：

```
service nginx start
```

接下来安装uwsgi：

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple uwsgi
```

查看uwsgi的版本，若显示则表示安装成功：

```
uwsgi --version
```

接下来我们创建uwsgi的配置文件，在命令行中输入：

```
vim main.ini
```

将以下内容输入到文本当中，其中wsgi-file为部署模型的python程序名，在文章之后会有程序的具体内容；socket为Nginx的转接地址；threads为同时开启的线程数，如需要同时调试多个模型，请增大线程数：

```
[uwsgi]
master = true
wsgi-file = main.py
callable = app
socket = 127.0.0.1:8001
processes = 4
threads = 2
buffer-size = 32768
```

全部配置完成后，运行只需要输入：

```
uwsgi main.ini
```

#### 4.4.3 图像分类模型部署

单片机运算内存较小，用其自带的运算器计算速度很慢，因此我们可以使用云服务器加持，从树莓派端收集图片，在树莓派端进行图片裁剪后发送给云服务器进行模型导入计算，并返回label值给树莓派。

##### 1.树莓派端图像裁剪

树莓派端通过调用opencv的摄像头函数采集图像后，进行图像缩放及图像均值化等简单操作，把图像缩放成224*224大小，并用直方图均值化的方法处理光照不均，最后通过端口发送图片给云服务器

```python
def load_image(img_path, size=(224, 224)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size)/255.0
    return img
```

##### 2.服务器端模型分类

云服务器端首先加载之前处理好的tflite模型文件，导入训练好的模型骨架和参数。

```python
def evaluate_model(interpreter, test_image):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    output = interpreter.tensor(output_index)
    output = np.argmax(output()[0])
    return output
```

通过8089这个端口号接收到图片后，将图片暂存之文件夹内，并读取该图片放入到预加载好的模型里

```python
interpreter = tf.lite.Interpreter(model_path='MobileNetV2.tflite')
interpreter.allocate_tensors()
```

此模型共分为30类

```python
classlist = ["apple", "banana", "blueberry", "cherry", "durian", "fig", "grape", "lemon", "litchi", "mango", "orange", "pineapple", "plum", "pomegranate", "strawberry", "aster", "begonia", "calla_lily", "chrysanthemum", "cornflower", "corydali", "dahlia", "daisy", "gentian", "mistflower", "nigella", "rose", "sandwort", "sunflower", "veronica"]
```

每次计算都会得到一个0到29的索引值，云服务器会根据索引值索引到类别，返回字符串给树莓派端。

```python
@app.route('/', methods=['post'])
def predict():
    upload_file = request.files['file']
    file_name = upload_file.filename
    file_path = '/home/ubuntu/inifyy/img'
    if upload_file:
        file_paths = os.path.join(file_path, file_name)
        upload_file.save(file_paths)
    test = load_image(file_paths)
    result = evaluate_model(interpreter, test)
    result = classlist[result]
    return result
```

#### 4.4.4 目标检测模型部署

##### 1.树莓派端数据发送

树莓派端首先用OpenCV包进行拍照与图像处理，接着利用requests模块发送post请求，并等待服务器返回运行结果，具体的部署代码如下：

```python
import requests
import numpy as np
import cv2


# 服务器公网地址
url = "http://127.0.0.1:8088/"
# post图片格式
content_type = 'image/jpeg'
headers = {'content-type': content_type}
# 定义摄像头
capture = cv2.VideoCapture(0)
while True:
    # 拍照与图片预处理
    ret, frame = capture.read()
    frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_CUBIC)
    # 将图片数据编码并发送
    img_encoded = cv2.imencode('.jpg', frame)[1]
    imgstring = np.array(img_encoded).tobytes()
    response = requests.post(url, data=imgstring, headers=headers)
    imgstring = np.asarray(bytearray(response.content), dtype="uint8")
    # 展示返回结果
    img = cv2.imdecode(imgstring, cv2.IMREAD_COLOR)
    cv2.imshow("video", img)
    c = cv2.waitKey(20)
    # 如果按q键，则终止
    if c == 113:
        break
cv2.destroyAllWindows()
```

##### 2.服务器端模型检测

服务器端的模型部署与之前在树莓派上部署模型类似，都用到了Tensorflow Object Detection的API。

首先我们需要在树莓派上下载Tensorflow Object Detection的API包，在树莓派命令行中输入：

```
git clone https://github.com/tensorflow/models
```

克隆完成后，将克隆的仓库进行重命名：

```
mv models-master models
```

下载目标检测API必要的软件包：

```
pip3 install tf_slim
pip3 install lvis
```

导入python的环境路径：

```
export PYTHONPATH=$PYTHONPATH:models/research/:models
```

接下来我们便可以进行目标检测模型的部署了，具体部署代码如下：

```python
import tensorflow as tf
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder
import cv2
from flask import Flask, request
app = Flask(__name__)


# 定义检测函数
def detect(interpreter, input_tensor):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  preprocessed_image, shapes = detection_model.preprocess(input_tensor)
  interpreter.set_tensor(input_details[0]['index'], preprocessed_image.numpy())
  interpreter.invoke()
  boxes = interpreter.get_tensor(output_details[0]['index'])
  classes = interpreter.get_tensor(output_details[1]['index'])
  scores = interpreter.get_tensor(output_details[2]['index'])
  return boxes, classes, scores


# 模型识别种类个数
num_classes = 16
# 模型位置
pipeline_config = 'pipeline.config'
# 模型标签
category_index = {1: {'id': 1, 'name': 'apple'}, 2: {'id': 2, 'name': 'banana'}, 3: {'id': 3, 'name': 'grape'}, 4: {'id': 4, 'name': 'kiwifruit'}, 5: {'id': 5, 'name': 'mango'}, 6: {'id': 6, 'name': 'orange'}, 7: {'id': 7, 'name': 'pear'}, 8: {'id': 8, 'name': 'stawberry'}, 9: {'id': 9, 'name': 'calla lily'}, 10: {'id': 10, 'name': 'cornflower'}, 11: {'id':11, 'name': 'corydalis'}, 12: {'id': 12, 'name': 'dahlia'}, 13: {'id': 13, 'name': 'daisy'}, 14: {'id': 14, 'name': 'gentian'}, 15: {'id': 15, 'name': 'nigella'}, 16: {'id': 16, 'name': 'sunflower'}}

# 定义模型
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
model_config.ssd.num_classes = num_classes
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(model_config=model_config, is_training=True)

# 加载tflite文件
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
label_id_offset = 1


# 定义预测函数，用于接受post及预测
@app.route('/', methods=["post"])
def predict():
    # 解码接收的图像文件
    imgstring = np.asarray(bytearray(request.data), dtype="uint8")
    img = cv2.imdecode(imgstring, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test = np.expand_dims(frame, axis=0)
    # 目标检测
    input_tensor = tf.convert_to_tensor(test, dtype=tf.float32)
    boxes, classes, scores = detect(interpreter, input_tensor)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        test[0],
        boxes[0],
        classes[0].astype(np.uint32) + label_id_offset,
        scores[0],
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    #返回运算结果
    frame = cv2.cvtColor(test[0], cv2.COLOR_BGR2RGB)
    img_encoded = cv2.imencode('.jpg', frame)[1]
    imgstring = np.array(img_encoded).tobytes()
    return imgstring

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8088)
```