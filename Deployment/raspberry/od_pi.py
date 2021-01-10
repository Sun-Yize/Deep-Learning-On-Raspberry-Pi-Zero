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