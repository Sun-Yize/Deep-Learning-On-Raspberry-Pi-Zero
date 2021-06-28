## 三、目标检测模型

### 3.1 数据集的选取

在目标检测过程中，我们从图像分类的30种分类中，选取了8种水果，8种花卉（共16种），作为我们的目标检测模型训练数据。具体的选取种类如下：

| 序号 |   英文名   | 中文名 |      | 序号 |   英文名   | 中文名 |
| :--: | :--------: | :----: | ---- | :--: | :--------: | :----: |
|  1   |   apple    |  苹果  |      |  9   | calla lily | 马蹄莲 |
|  2   |   banana   |  香蕉  |      |  10  | cornflower | 矢车菊 |
|  3   |   grape    |  葡萄  |      |  11  | corydalis  | 延胡索 |
|  4   | kiwifruit  | 猕猴桃 |      |  12  |   dahlia   | 大丽花 |
|  5   |   mango    |  芒果  |      |  13  |   daisy    |  雏菊  |
|  6   |   orange   |  橘子  |      |  14  |  gentian   |  龙胆  |
|  7   |    pear    |   梨   |      |  15  |  nigella   | 黑种草 |
|  8   | strawberry |  草莓  |      |  16  | sunflower  | 向日葵 |


### 3.2 数据集获取

在目标检测领域，imagenet提供了丰富的目标检测原始数据，包括了不同种类的花卉和水果。我们在imagenet官网上，下载目标检测所需的数据。

+ 点击imagenet官网：http://www.image-net.org/

+ 搜索我们需要的数据，这里我们以香蕉banana为例：

<img src="./image/img301.png" width=70%>

+ 点击Downloads，分别下载原始图片以及边界框。

<img src="./image/img302.png" width=70%>

下载好后，我们可以进一步查看我们下载的目标检测数据。LabelImg是目标检测标记图像的工具，它既可以用于标注图片，也可以用于查看目标检测的数据。我们可以在github中下载其最新版本，其github页面上有关于如何安装和使用它的非常清晰的说明。

[LabelImg说明文档](https://github.com/tzutalin/labelImg)

[点击下载LabelImg](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1)

我们将我们下载的images与Bounding Boxes放入同一个文件夹，并用LabelImg打开，具体效果如下：

<img src="./image/img303.png" width=70%>

同时我们也可以自己用LabelImg进行数据标注，LabelImg保存一个.xml文件，其中包含每个图像的标签数据。这些.xml文件将用于生成TFRecords，它们是TensorFlow训练的输入之一。

### 3.3 目标检测模型选择

在当前的目标检测领域中，已有较为成熟的研究成果。目前目标检测主要分为两个领域

（1）**two-stage方法**，如R-CNN系算法，其主要思路是先通过启发式方法或者CNN网络（RPN)产生一系列稀疏的候选框，然后对这些候选框进行分类与回归，two-stage方法的优势是准确度高

（2）**one-stage方法**，如Yolo和SSD，其主要思路是均匀地在图片的不同位置进行密集抽样，抽样时可以采用不同尺度和长宽比，然后利用CNN提取特征后直接进行分类与回归，整个过程只需要一步，所以其优势是速度快，但是均匀的密集采样的一个重要缺点是训练比较困难，这主要是因为正样本与负样本极其不均衡，导致模型准确度稍低。

因为我们在之后部署到树莓派的过程中，计算能力有限，所以这里我们选择one-stage方法中的SSD，在移动端仍可以保持较快的运算速度。

同时，我们选择将SSD与MobileNet相结合，生成以MobileNet为基底的SSD-MobileNetV2网络。具体的网络结构如下：

<img src="./image/img304.png" width=100%>

### 3.4 Tensorflow2目标检测API

TensorFlow目标检测API是一个基于TensorFlow的开源框架，可轻松构建，训练和部署对象检测模型。在2020年6月，TensorFlow更新的目标检测，并适用于了TensorFlow2的版本。在本教程内，我们将基于TF2来进行目标检测。

首先我们要从github上克隆完整的Tensorflow object detection仓库，在命令行输入如下命令：

```
git clone https://github.com/tensorflow/models
```

克隆或下载完成后，将models-master重命名为models。

接下来，我们的主要操作全部在 models/research/object_detection 中进行。

同时，我们也可以下载一些预训练好的模型，用于之后模型的训练。我们在TensorFlow目标检测的仓库中可以找到对应的模型，在如下地址下载：

[TF2预训练模型](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

在本篇文章中，我们主要使用了SSD-MobileNetV2模型，所以点击SSD-MobileNetV2模型进行下载：

[SSD MobileNet v2 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)

### 3.5 配置Tensorflow2环境

首先，我们需要使用pip安装tensorflow2版本环境。我们在命令行中输入：

```
pip install tensorflow==2.3.0
```

如果使用的是GPU设备，则安装对应的tensorflow-gpu版本，在安装的同时，系统会自动安装对应的CUDA和cuDNN。

等待一段时间安装好后，我们再安装其他必要的软件包，在命令行中输入：

```
pip install tf_slim
pip install lvis
```

当软件包安装好后，我们需要编译tensorflow目标检测API中的Protobufs，首先切换到对应目录，然后再用protoc命令进行编译：

```
cd models/research
protoc object_detection/protos/*.proto --python_out=.
```

最后我们还需要配置PYTHONPATH环境变量，输入如下命令进行PYTHONPATH环境变量的配置：

```
export PYTHONPATH=$PYTHONPATH:models/research/:models
```

完成TensorFlow对象检测API的所有设置并可以使用了。

### 3.6 生成TFRecords格式数据

接下来我们要生成TFRecords格式的数据，首先保证目标检测的数据已经处理为了如下格式，在文件夹中排列：

<img src="./image/img305.png" width=70%>

在models/research/object_detection中新建images文件夹。

将这些图片中，以10:1的比例分别放入models/research/object_detection/train，models/research/object_detection/test，用于后续的处理。

处理好图像后，便开始生成TFRecords了，该记录用作TensorFlow训练模型的输入数据。我们使用本教程github仓库中的xml_to_csv.py和generate_tfrecord.py文件。 

首先，图像.xml数据将用于创建.csv文件，其中包含了训练集和测试集的所有数据。首先打开object_detection文件夹，在命令行中运行此命令：

```
python xml_to_csv.py
```

这将在object_detection/images文件夹中创建train_labels.csv和test_labels.csv文件。

接下来，在文本编辑器中打开generate_tfrecord.py文件。并用generate_tfrecord.py生成对应的TFRecords文件，执行以下命令：

```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=training/train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=training/test.record
```

执行完命令后，我们便发现在object_detection/training文件夹中有了train.record和test.record文件，这两个文件分别作为之后目标检测的训练集与测试集使用。

### 3.7 目标检测模型训练

在执行训练之前，我们先介绍一下训练所需的各个文件对应的含义。

#### 3.7.1 label_map.pbtxt标签图

标签图通过定义分类名称和对应的分类编号的映射，来说明训练中每个编号对应的内容是什么。我们将标签图放在models/research/object_detection/training文件夹中，并按照对应的格式写好，具体的格式如下：

```
item {
  id: 1
  name: 'apple'
}

item {
  id: 2
  name: 'banana'
}

item {
  id: 3
  name: 'grape'
}

...
```

标签映射ID编号应与generate_tfrecord.py文件中定义的编号相同。

#### 3.7.2 pipeline.config配置

我们在训练不同的目标检测模型时，需要对配置文件进行修改，以下是在训练SSD-MobileNetV2模型时所用到的配置文件，具体如下：

```
model {
  ssd {
    inplace_batchnorm_update: true
    freeze_batchnorm: false
    num_classes: 16
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    encode_background_as_zeros: true
    anchor_generator {
      ssd_anchor_generator {
        num_layers: 6
        min_scale: 0.2
        max_scale: 0.95
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.3333
      }
    }
    image_resizer {
      fixed_shape_resizer {
        height: 300
        width: 300
      }
    }
    box_predictor {
      convolutional_box_predictor {
        min_depth: 0
        max_depth: 0
        num_layers_before_predictor: 0
        use_dropout: false
        dropout_keep_probability: 0.8
        kernel_size: 1
        box_code_size: 4
        apply_sigmoid_to_scores: false
        class_prediction_bias_init: -4.6
        conv_hyperparams {
          activation: RELU_6,
          regularizer {
            l2_regularizer {
              weight: 0.00004
            }
          }
          initializer {
            random_normal_initializer {
              stddev: 0.01
              mean: 0.0
            }
          }
          batch_norm {
            train: true,
            scale: true,
            center: true,
            decay: 0.97,
            epsilon: 0.001,
          }
        }
      }
    }
    feature_extractor {
      type: 'ssd_mobilenet_v2_keras'
      min_depth: 16
      depth_multiplier: 1.0
      conv_hyperparams {
        activation: RELU_6,
        regularizer {
          l2_regularizer {
            weight: 0.00004
          }
        }
        initializer {
          truncated_normal_initializer {
            stddev: 0.03
            mean: 0.0
          }
        }
        batch_norm {
          train: true,
          scale: true,
          center: true,
          decay: 0.97,
          epsilon: 0.001,
        }
      }
      override_base_feature_extractor_hyperparams: true
    }
    loss {
      classification_loss {
        weighted_sigmoid_focal {
          alpha: 0.75,
          gamma: 2.0
        }
      }
      localization_loss {
        weighted_smooth_l1 {
          delta: 1.0
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    normalize_loss_by_num_matches: true
    normalize_loc_loss_by_codesize: true
    post_processing {
      batch_non_max_suppression {
        score_threshold: 1e-8
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
  }
}

train_config: {
  fine_tune_checkpoint_version: V2
  fine_tune_checkpoint: "models/research/object_detection/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
  fine_tune_checkpoint_type: "detection"
  batch_size: 256
  sync_replicas: true
  startup_delay_steps: 0
  replicas_to_aggregate: 8
  num_steps: 100000
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    ssd_random_crop {
    }
  }
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: .8
          total_steps: 50000
          warmup_learning_rate: 0.13333
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
}

train_input_reader: {
  label_map_path: "models/research/object_detection/training/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "models/research/object_detection/training/train.record"
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}

eval_input_reader: {
  label_map_path: "models/research/object_detection/training/label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "models/research/object_detection/training/test.record"
  }
}
```

该文件基于Tensorflow Object Detection自带的配置文件进行修改而得出，如果需要自己训练模型，可以models/research/object_detection/configs/tf2中找到对应的模型，一般对应的配置文件主要修改以下几处：

- 将num_classes更改为要分类器检测的不同对象的数量，本文修改的是16。
- 将fine_tune_checkpoint更改为预训练模型的对应路径，如果没有预训练模型，则删除本行。
- 将fine_tune_checkpoint_type修改为: "detection"，如果没有预训练模型，则删除本行。
- 在train_input_reader部分中，将input_path和label_map_path更改为：
  - input_path：“ models/research/object_detection/training/train.record”
  - label_map_path：“ models/research/object_detection/training/label_map.pbtxt”
- 在eval_input_reader部分中，将input_path和label_map_path更改为：
  - input_path：“models/research/object_detection/training/test.record”
  - label_map_path：“models/research/object_detection/training/label_map.pbtxt”

#### 3.7.3 开始训练

Tensorflow2训练时，主要使用object_detection文件夹下的model_main_tf2.py文件进行数据集的训练。

model_main_tf2.py文件主要有如下几个输入的选项：

+ pipeline_config_path：输入对应的配置文件位置。
+ model_dir：训练模型时对应的文件夹位置。
+ checkpoint_every_n：每n步对模型进行一次保存
+ record_summaries：储存模型的训练过程

我们使用如下命令开始对模型进行训练：

```
python3 model_main_tf2.py \
  --logtostderr \
  --model_dir=training \
  --pipeline_config_path=training/pipline.config \
  --checkpoint_every_n=200 \
  --record_summaries=training
```

在执行命令后，TensorFlow将初始化训练。初始化过程可能需要1-2分钟左右。当初始化完成后，程序便开始进行正式训练：

<img src="./image/img306.png" width=100%>

在训练过程中，会从命令行输出每100epoach的训练结果。在本文所部署的SSD-MobileNetV2模型，使用了NVIDIA Tesla V100显卡进行训练，训练约5-6小时后，结果开始逐步收敛，损失函数最终收敛到0.1以下，目标检测模型可以基本实现正确的检测。

### 3.8 TensorBoard查看训练进度

TensorBoard可以用于实时查看训练的进度如何，我们可以在命令行或jupyter notebook中调用TensorBoard，实时查看损失函数的下降情况以及训练的具体进度，这里我们用jupyter notebook进行查看，具体操作如下：

首先导入TensorBoard：

```python
%load_ext tensorboard

from tensorboard import notebook
notebook.list() 
```

接着输入训练对应文件夹，启用TensorBoard：

```python
notebook.start("--logdir models/research/object_detection/training")
```

我们可以在输出界面看到训练的具体情况：

<img src="./image/img307.png" width=100%>

### 3.9 导出训练模型

接下来我们导出训练好的模型，并将模型转换为tflite格式。

#### 3.9.1 导出.pb格式模型

我们使用object_detection文件夹中的export_tflite_graph_tf2.py文件，先将模型导出：

```
python export_tflite_graph_tf2.py \
  --pipeline_config_path training/pipeline.config \
  --trained_checkpoint_dir training \
  --output_directory trainingModel
```

导出的模型位于object_detection/trainingModel文件夹，如果不需要部署到单片机或其他移动端设备，则可以对此模型直接使用。

#### 3.9.2 .pb格式模型转化为.tflite

tflite是为了将深度学习模型部署在移动端和嵌入式设备的工具包，可以把训练好的模型通过转化、部署和优化三个步骤，达到提升运算速度，减少内存、显存占用的效果。

我们需要对已有的模型进行转换，首先使用pip安装tf-nightly：

```
pip install tf-nightly
```

tf-nightly是支持tensorflow2的软件包，并首次被添加到tensorflow 2.3版本。

接下来我们使用tflite_convert对模型进行转换，将原有的模型转化为.tflite格式，在命令行中输入如下命令：

```
tflite_convert --saved_model_dir=trainingModel/saved_model --output_file=trainingModel/model.tflite
```

这时我们可以看到trainingModel文件夹中生成了model.tflite文件，此时我们便可以用此模型部署到树莓派上了。