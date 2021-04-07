# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 上午11:13
# @Author  : red0orange
# @File    : testGPIO.py
# @Content : 测试GPIO的功能
import RPi.GPIO as GPIO
import time
import signal
import atexit

if __name__ == '__main__':
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(17, GPIO.OUT)
    # # GPIO.output(17, GPIO.HIGH)
    # while True:
    #     a = input()
    #     if a == '0':
    #         GPIO.output(17, GPIO.LOW)
    #     elif a == '1':
    #         GPIO.output(17, GPIO.HIGH)

    atexit.register(GPIO.cleanup)

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT, initial=False)
    p = GPIO.PWM(18, 50)  # 50HZ
    p.start(0)
    time.sleep(1)

    while True:
        p.ChangeDutyCycle(6)  # 90度
        time.sleep(1.5)
        p.ChangeDutyCycle(11)  # 0度
        time.sleep(1.5)
