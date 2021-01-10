import tensorflow as tf
import numpy as np

### 云服务器端加载之前处理好的tflite模型文件，导入训练好的模型骨架和参数
def evaluate_model(interpreter, test_image):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    output = interpreter.tensor(output_index)
    output = np.argmax(output()[0])
    return output

interpreter = tf.lite.Interpreter(model_path='MobileNetV2.tflite')
interpreter.allocate_tensors()
### 此模型共分为30类
classlist = ["apple", "banana", "blueberry", "cherry", "durian", "fig", "grape", "lemon", "litchi", "mango", "orange", "pineapple", "plum", "pomegranate", "strawberry", "aster", "begonia", "calla_lily", "chrysanthemum", "cornflower", "corydali", "dahlia", "daisy", "gentian", "mistflower", "nigella", "rose", "sandwort", "sunflower", "veronica"]

#### 每次计算都会得到一个0到29的索引值，云服务器会根据索引值索引到类别，返回字符串给树莓派端
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

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8087)