#!/usr/bin/python3
# coding=utf8
import sys

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
import cv2
import numpy as np
import time
import math
import signal
import threading
import PWMServo
import check_camera
import timeout_decorator
from config import *
from cv_ImgAddText import *
import Serial_Servo_Running as SSR

PWMServo.setServo(1, 1500, 500)
PWMServo.setServo(2, 1500, 500)

debug = True
# 阈值
conf_threshold = 0.6

# 模型位置
modelFile = "/home/pi/human_code/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "/home/pi/human_code/models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

servo2_pulse = servo2

@timeout_decorator.timeout(0.5, use_signals=False)
def Camera_isOpened():
    global stream, cap
    cap = cv2.VideoCapture(stream)


# 摄像头默认分辨率640x480,处理图像时会相应的缩小图像进行处理，这样可以加快运行速度
# 缩小时保持比例4：3,且缩小后的分辨率应该是整数
c = 80
width, height = c * 4, c * 3
resolution = str(width) + "x" + str(height)
orgFrame = None
Running = True
ret = False
stream = "http://127.0.0.1:8080/?action=stream?dummy=param.mjpg"
try:
    Camera_isOpened()
    cap = cv2.VideoCapture(stream)
except:
    print('Unable to detect camera! \n')
    check_camera.CheckCamera()

orgFrame = None
ret = False
Running = True


def get_image():
    global orgFrame
    global ret
    global Running
    global stream, cap
    global width, height
    while True:
        if Running:
            try:
                if cap.isOpened():
                    ret, orgFrame = cap.read()
                else:
                    time.sleep(0.01)
            except:
                cap = cv2.VideoCapture(stream)
                print('Restart Camera Successful!')
        else:
            time.sleep(0.01)


th1 = threading.Thread(target=get_image)
th1.setDaemon(True)
th1.start()

d_pulse = 10
start_greet = False
action_finish = True

def runAction():
    global start_greet
    global action_finish
    global d_pulse, servo2_pulse
    while True:
        if start_greet:
            start_greet = False
            action_finish = False
            PWMServo.setServo(1, 1800, 200)
            time.sleep(0.2)
            PWMServo.setServo(1, 1200, 200)
            time.sleep(0.2)
            PWMServo.setServo(1, 1800, 200)
            time.sleep(0.2)
            PWMServo.setServo(1, 1200, 200)
            time.sleep(0.2)
            PWMServo.setServo(1, 1500, 100)
            SSR.change_action_value('pick', 1)
            time.sleep(1)
            action_finish = True
        else:
            if servo2_pulse > 2000 or servo2_pulse < 1000:
               d_pulse = -d_pulse
               servo2_pulse += d_pulse
               PWMServo.setServo(2, servo2_pulse, 50)
               time.sleep(0.05)

# else:
#   time.sleep(0.01)
th2 = threading.Thread(target=runAction)
th2.setDaemon(True)
th2.start()
SSR.runAction('0')
while True:
    if orgFrame is not None and ret:
        if Running:
            t1 = cv2.getTickCount()
            orgframe = cv2.resize(orgFrame, (width, height), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
            frame = cv2.GaussianBlur(orgframe, (3, 3), 0)  # 高斯模糊
            Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # 将图片转换到LAB空间
        # 人脸识别
            blob = cv2.dnn.blobFromImage(orgframe, 1, (150, 150), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()  # 计算识别
            for i in range(detections.shape[2]):
              confidence = detections[0, 0, i, 2]
              if confidence > conf_threshold:
                # 识别到的人了的各个坐标转换会未缩放前的坐标
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                cv2.rectangle(orgframe, (x1, y1), (x2, y2), (0, 255, 0), 2, 8)  # 将识别到的人脸框出
                if abs((x1 + x2) / 2 - width / 2) < width / 4:
                  if action_finish:
                      start_greet = True

            t2 = cv2.getTickCount()
            time_r = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / time_r 
            if debug:
                orgframe = cv2ImgAddText(orgframe, "人脸识别", 10, 10, textColor=(0, 0, 0), textSize=20)
                cv2.putText(orgframe, "FPS:" + str(int(fps)),
                            (10, orgframe.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255),
                            2)  # (0, 0, 255)BGR
                cv2.namedWindow('orgframe')
                cv2.imshow("orgframe", orgframe)
                cv2.waitKey(1)
        else:
            time.sleep(0.01)
    else:
        time.sleep(0.01)
cap.release()
cv2.destroyAllWindows()