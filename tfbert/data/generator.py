# -*- coding: UTF-8 -*-
__author__ = 'huanghui'
__date__ = '2021/4/16 22:51'
__project__ = 'tfbert'

import copy

import numpy as np
from typing import List, Dict, Optional, Union
import random


def sequence_padding(ids: List, max_length=None, pad_id=0, mode='post'):
    """
    copy的苏神sequence_padding代码
    :param ids:
    :param max_length:
    :param pad_id:
    :param mode:
    :return:
    """
    if not isinstance(ids[0], list):
        return ids
    if max_length is None:
        max_length = max([len(x) for x in ids])

    pad_width = [(0, 0) for _ in np.shape(ids[0])]
    outputs = []
    for id_ in ids:
        x = id_[:max_length]
        if mode == 'post':
            pad_width[0] = (0, max_length - len(x))
        elif mode == 'pre':
            pad_width[0] = (max_length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=pad_id)
        outputs.append(x)

    return np.array(outputs)


def collate_batch(
        examples: Union[Dict, List[Dict]],
        max_length=None,
        pad_id: Optional[Union[int, Dict]] = 0,
        mode='post'):
    """
    :param examples: 可以是一个二维列表，可以是一个元素为字典的列表， 也可以是一个字典
    :param max_length:
    :param pad_id: 单个id或者字典指定，字典指定的话会对每一个字典填充对应的id
    :param mode:
    :return:
    """
    if isinstance(examples, dict) or isinstance(examples[0], dict):
        if isinstance(examples, dict):
            result = examples
        else:
            result = {k: [] for k in examples[0]}
            for i in range(len(examples)):
                for k in result:
                    result[k].append(examples[i][k])
        if not isinstance(pad_id, dict):
            pad_id = {k: pad_id for k in result}
        for k, v in result.items():
            if isinstance(v[0], list):
                result[k] = sequence_padding(v, max_length, pad_id[k], mode)
            else:
                result[k] = v
        return result
    elif isinstance(examples[0], list):
        return sequence_padding(examples, max_length, pad_id, mode)


class SimpleDataset:
    def __init__(
            self,
            examples: List[Dict],
            is_training=False,
            batch_size=1,
            drop_last=False,
            padding=False,
            max_length: Optional[int] = None,
            pad_id: Optional[Union[int, Dict]] = 0,
            pad_mode='post',
            padding_all=False):
        """
        简单的dataset包装
        :param examples: 列表，元素为键值相同的字典
        :param is_training: 是否训练，训练模式会对数据shuffle
        :param batch_size: 批次大小
        :param drop_last: 舍弃最后一个不足批次大小的数据
        :param padding: 是否填充
        :param max_length: 填充的最大，为None的话补全到当前batch最大长度
        :param pad_id: 填充id，可以是单个id，也可以是字典类型，每个字典对应自己的填充id
        :param pad_mode: post 或 pre，即在后边填充和在前边填充
        :param padding_all:是否直接对所有examples进行padding，这样会在迭代过程中减少padding的操作
        """
        if is_training:
            random.shuffle(examples)

        self.examples = {}
        self.columns = []
        for i in range(len(examples)):
            for k, v in examples[i].items():
                if i == 0:
                    self.examples[k] = []
                    self.columns.append(k)
                self.examples[k].append(v)
        self.back_columns = copy.deepcopy(self.columns)
        if not isinstance(pad_id, dict):
            pad_id = {k: pad_id for k in self.examples}
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.idx = 0
        self.padding = padding
        self.pad_id = pad_id
        self.max_length = max_length
        self.pad_mode = pad_mode

        self.padded = False
        if padding_all:
            self.examples = collate_batch(self.examples, self.max_length, self.pad_id, self.pad_mode)
            self.padded = True

    def __num_batch__(self):
        num_features = len(self.examples[self.columns[0]])
        if self.drop_last:
            num_batch = num_features // self.batch_size
        else:
            num_batch = (num_features + self.batch_size - 1) // self.batch_size
        return num_batch

    @property
    def num_batch(self):
        return self.__num_batch__()

    def __len__(self):
        return self.num_batch

    def __getitem__(self, item):
        if item in self.columns:
            return self.examples[item]
        elif isinstance(item, int):
            data = {}
            for k in self.columns:
                data[k] = self.examples[k][item]
        else:
            raise ValueError(f"type error of {item}")
        return data

    def __iter__(self):
        return self

    def __repr__(self):
        return f"SimpleDataset({{\n    features: {self.columns},\n    num_batch: {self.num_batch}\n}})"

    def __next__(self):
        if self.idx < self.num_batch:
            start = self.idx * self.batch_size
            end = (self.idx + 1) * self.batch_size
            batch = {}
            for k in self.columns:
                if not self.padded and self.padding:
                    batch[k] = sequence_padding(self.examples[k][start: end], self.max_length, self.pad_id[k],
                                                self.pad_mode)
                else:
                    batch[k] = self.examples[k][start: end]
            self.idx += 1
            return batch
        else:
            self.idx = 0
            raise StopIteration

    def remove_columns(self, remove_columns):
        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]
        elif not isinstance(remove_columns, list):
            remove_columns = []
        for remove_column in remove_columns:
            if remove_column in self.examples:
                self.examples.pop(remove_column)
                self.columns.remove(remove_column)
                self.back_columns.remove(remove_column)

    def format_as(self, columns):
        if not isinstance(columns, list):
            columns = [columns]
        new_columns = []
        for column in columns:
            if column in self.back_columns:
                new_columns.append(column)
        self.columns = new_columns

    def restore_columns(self):
        self.columns = copy.deepcopy(self.back_columns)

    def get_all_features(self):
        if self.padded:
            return self.examples
        return collate_batch(self.examples, self.max_length, self.pad_id, self.pad_mode)
