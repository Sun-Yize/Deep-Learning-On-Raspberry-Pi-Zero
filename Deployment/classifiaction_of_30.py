import cv2
import numpy as np
import tensorflow as tf

#### 加载图像并裁剪到224*224的大小
def load_image(img_path, size = (224,224)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size)/255.0
    return img

### 加载模型函数
def evaluate_model(interpreter, test_image):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    output = interpreter.tensor(output_index)
    output = np.argmax(output()[0])
    return output

#### 读取模型文件
interpreter = tf.lite.Interpreter(model_path='MobileNetV2.tflite')
interpreter.allocate_tensors()
classlist = ["apple", "banana", "blueberry", "cherry", "durian", "fig", "grape", "lemon", "litchi", "mango", "orange", "pineapple", "plum", "pomegranate", "strawberry", "aster", "begonia", "calla_lily", "chrysanthemum", "cornflower", "corydali", "dahlia", "daisy", "gentian", "mistflower", "nigella", "rose", "sandwort", "sunflower", "veronica"]

#### 通过opencv包打开摄像头进行拍摄
capture = cv2.VideoCapture(0)#0为树莓派的内置摄像头
while(True):
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    cv2.imwrite("temp.jpg", frame)
    test = load_image("temp.jpg")
    # 用模型进行识别
    result = evaluate_model(interpreter, test)
    print(classlist[result])
    # 呈现识别的结果
    cv2.imshow("video", frame)
    c = cv2.waitKey(100)
    # 如果按q键，则终止
    if c == 113:
        break
cv2.destroyAllWindows()