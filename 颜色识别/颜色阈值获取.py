# -*- coding:utf-8 -*-

import cv2
import numpy as np

"""
功能：读取一张图片，显示出来，转化为HSV色彩空间
     并通过滑块调节HSV阈值，实时显示
"""

image = cv2.imread('E:/14197/graduation design/opencv/image/22.jpg')  # 根据路径读取一张图片，opencv读出来的是BGR模式
cv2.imshow("BGR", image)  # 显示图片

Lab_low = np.array([0, 0, 0])
Lab_high = np.array([0, 0, 0])

def L_low(value):
    Lab_low[0] = value

def L_high(value):
    Lab_high[0] = value

def A_low(value):
    Lab_low[1] = value

def A_high(value):
    Lab_high[1] = value

def B_low(value):
    Lab_low[2] = value

def B_high(value):
    Lab_high[2] = value

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

cv2.createTrackbar('L low', 'image', 0, 255, L_low)
cv2.createTrackbar('L high', 'image', 0, 255, L_high)
cv2.createTrackbar('A low', 'image', 0, 255, A_low)
cv2.createTrackbar('A high', 'image', 0, 255, A_high)
cv2.createTrackbar('B low', 'image', 0, 255, B_low)
cv2.createTrackbar('B high', 'image', 0, 255, B_high)

while True:
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # BGR转Lab
    dst = cv2.inRange(dst, Lab_low, Lab_high)  # 通过Lab的高低阈值，提取图像部分区域
    cv2.imshow('dst', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()