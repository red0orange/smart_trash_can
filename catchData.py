# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 上午11:12
# @Author  : red0orange
# @File    : catchData.py
# @Content : 一个调用摄像头采集图像的脚本
import cv2
import os

if __name__ == '__main__':
    savePath = 'new_data'

    cap = cv2.VideoCapture(0)
    c = 0
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('window', cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)))
        k = cv2.waitKey(1)
        if k == ord('c'):
            path = os.path.join(savePath, str(c))
            os.makedirs(path, exist_ok=True)
            print(os.path.join(path, str(i) + '.jpg'))
            cv2.imwrite(os.path.join(path, str(i) + '.jpg'), frame)
            i += 1
        elif k == ord('n'):
            print('next_class: ', c + 1)
            c += 1
            i = 0
