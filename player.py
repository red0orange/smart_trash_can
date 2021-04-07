# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 上午10:50
# @Author  : red0orange
# @File    : player.py
# @Content : 播放声音的调用类
from omxplayer.player import OMXPlayer


class Player:
    def __init__(self):
        self.player = None
        self.playing = None
        pass

    def playerExit(self, code):
        print('exit', code)
        self.playing = False

    def playFile(self, file, volume):
        if self.player is None:
            self.player = OMXPlayer(file)
            self.player.set_volume(volume)
            self.player.exitEvent += lambda _, exit_code: self.playerExit(exit_code)
        else:
            self.player.load(file)
        print('Playing:', file)
        self.playing = True

    def quitPlayer(self):
        if self.player != None:
            self.player.quit()
