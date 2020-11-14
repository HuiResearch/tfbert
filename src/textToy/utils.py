# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: utils.py
@date: 2020/09/12
"""
import time
import random
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib


def set_seed(seed):
    '''
    随机种子设置，发现GPU上没啥用
    :param seed:
    :return:
    '''
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def gpu_is_available():
    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    if not num_gpus:
        return False
    return True


def get_devices():
    local_device_protos = device_lib.list_local_devices()
    gpus = [d.name for d in local_device_protos if d.device_type == 'GPU']
    if not gpus:
        return ['/device:CPU:0']
    return gpus


# 自定义进度条工具
class ProgressBar:
    """
    自定义进度条工具，避免tqdm在win10下不能单行打印的问题，
    但貌似如果嵌套进度条无法打印第一个，第一个进度条会被后边的掩盖
    """

    def __init__(self, iterator, total=None, max_length=50, desc="", bar="█"):
        if total is not None:
            self.total = total
        else:
            if hasattr(iterator, "__len__"):
                self.total = len(iterator)
            else:
                self.total = None
        self.max_len = max_length
        self.iterator = iterator
        self.desc = desc
        self.bar = bar
        self.count = 0
        self.start_time = time.time()
        self.last_time = self.start_time

    def __iter__(self):
        for el in self.iterator:
            yield el
            self.count += 1
            self.last_time = time.time()
            self.refresh()

    @classmethod
    def format_time(cls, seconds):
        mins, s = divmod(int(seconds), 60)
        h, m = divmod(mins, 60)
        if h:
            return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
        else:
            return '{0:02d}:{1:02d}'.format(m, s)

    def format_msg(self):

        msg = "\r"
        if self.desc:
            msg += self.desc + ":  "

        if self.total is not None:
            percent = int(self.count / self.total * 100)
            length = int(self.count / self.total * self.max_len)
            msg += "{}%".format(percent)
            msg += "|" + self.bar * length + " " * (
                    self.max_len - length) + "| " + "{}/{}".format(self.count, self.total)
        else:
            msg += "{} item".format(self.count)

        seconds = self.last_time - self.start_time
        if seconds != 0:
            speed = self.count / seconds

            if self.total is not None:
                remain = (self.total - self.count) / speed
                msg += " [{} : {},  {:.2f} item/s]".format(self.format_time(seconds),
                                                           self.format_time(remain), speed)
            else:
                msg += " [{},  {:.2f} item/s]".format(self.format_time(seconds), speed)
        return msg

    def set_description(self, desc):
        self.desc = desc

    def refresh(self):
        end = ""
        if self.total is not None and self.count == self.total:
            end = "\n"
        print(self.format_msg(), end=end, sep='')
