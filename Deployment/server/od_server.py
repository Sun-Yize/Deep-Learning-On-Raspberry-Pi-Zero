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