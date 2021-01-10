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