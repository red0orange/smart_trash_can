# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 上午10:47
# @Author  : red0orange
# @File    : main.py
# @Content : 运行智能垃圾桶的主程序
import os
import atexit
import cv2
import numpy as np
import time

from detector import ONNXDetector
from player import Player

import RPi.GPIO as GPIO


if __name__ == '__main__':
    time.sleep(8)

    # TODO 打开语音提示
    # cVideo = '/home/pi/可回收.m4a'
    # ncVideo = '/home/pi/不可回收.m4a'
    # player = Player()

    # 初始化舵机的控制引脚GPIO
    atexit.register(GPIO.cleanup)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT, initial=False)
    p = GPIO.PWM(18, 50)  # 50HZ
    p.start(0)
    time.sleep(0.25)

    # 初始化图片检测者对象
    detector = ONNXDetector("/home/pi/trash/4_class_20.proto")

    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    # 初始化变量
    FindNum = 0
    nFindNum = 0
    ncNum = 0
    cNum = 0
    angle = 11
    # 开始主循环
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.clock()
            predict_score = detector.detect(frame)
            end = time.clock()
            print('predict time: {}'.format(end-start))

            print("class_id: {},  class_pro: {}".format(np.argmax(predict_score), np.max(predict_score)))
            items = {0: '电池', 1: '易拉罐', 2: '塑料瓶', 3: '纸团'}
            if np.max(predict_score) > 0.7:
                classNum = np.argmax(predict_score)
                if classNum == 1 or classNum == 2:
                    if FindNum >= 3:
                        if angle != 6:
                            angle = 6
                            p.ChangeDutyCycle(angle)
                        time.sleep(0.2)
                        FindNum = 0
                    else:
                        FindNum += 1
                else:
                    if classNum == 0 or classNum == 3:
                        # player.playFile(ncVideo,1)
                        if angle != 11:
                            angle = 11
                            p.ChangeDutyCycle(angle)
                            FindNum = 0
                        time.sleep(0.25)
                        # player.quit()
                        ncNum = 0
                    else:
                        ncNum += 1
                print(items[classNum])
            else:
                if nFindNum >= 8:
                    nFindNum = 0
                    if angle != 11:
                        print('11')
                        angle = 11
                        p.ChangeDutyCycle(angle)
                    time.sleep(0.2)
                else:
                    nFindNum += 1
                print('找不到目标')

            # cv2.imshow('test',frame)
            # cv2.waitKey(5)
        except:
            print("error run !")
            pass
