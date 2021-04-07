# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 上午10:51
# @Author  : red0orange
# @File    : detector.py
# @Content : 封装的识别类
import cv2
import time
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def pad(img, size):
    h, w, _ = img.shape
    max_len = max(h, w)
    result_img = np.zeros((max_len, max_len, 3), dtype=np.uint8)
    if h >= w:
        result_img[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        result_img[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    result_img = cv2.resize(result_img, size)
    return result_img


class ONNXDetector(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = cv2.dnn.readNetFromONNX(model_path)
        pass

    def detect(self, image):
        image = pad(image, (224, 224))

        image = np.asarray(image, dtype=np.float) / 255
        image = image.transpose(2, 0, 1)
        res = np.zeros_like(image, dtype=np.float)
        for i, (t, m, s) in enumerate(zip(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
            t = np.subtract(t, m)
            t = np.divide(t, s)
            res[i] = t
        image = res[np.newaxis, :]

        # img = np.full((224,224,3),fill_value=3,dtype=np.uint8)
        self.model.setInput(image)
        pro = self.model.forward()
        pro = softmax(pro.squeeze())
        return pro
