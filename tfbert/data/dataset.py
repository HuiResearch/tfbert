# -*- coding: UTF-8 -*-
__author__ = 'huanghui'
__date__ = '2021/4/16 22:51'
__project__ = 'tfbert'

import copy
from . import BaseClass, compute_shapes, compute_types, compute_types_and_shapes_from_dataset
import numpy as np
from typing import List, Dict, Optional, Union
import random
import tensorflow.compat.v1 as tf


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
        max_length: Optional[Union[int, Dict]] = None,
        pad_id: Optional[Union[int, Dict]] = 0,
        mode='post'):
    """
    :param examples: 可以是一个二维列表，可以是一个元素为字典的列表， 也可以是一个字典
    :param max_length:单个int或者字典指定，字典指定的话会对每一个字典填充对应的长度
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
        if not isinstance(max_length, dict):
            max_length = {k: max_length for k in result}
        for k, v in result.items():
            if isinstance(v[0], list):
                result[k] = sequence_padding(v, max_length[k], pad_id[k], mode)
            else:
                result[k] = v
        return result
    elif isinstance(examples[0], list):
        return sequence_padding(examples, max_length, pad_id, mode)


class Dataset:
    def __init__(
            self,
            features: List[Union[Dict, BaseClass]],
            is_training=False,
            batch_size=1,
            drop_last=False,
            buffer_size=100,
            padding=False,
            max_length: Optional[Union[int, Dict]] = None,
            pad_id: Optional[Union[int, Dict]] = 0,
            pad_mode='post',
            padding_all=False):
        """
        简单的dataset包装，未完成 map 方法
        :param features: 列表，元素为键值相同的字典
        :param is_training: 是否训练，训练模式会对数据shuffle
        :param batch_size: 批次大小
        :param drop_last: 舍弃最后一个不足批次大小的数据
        :param padding: 是否填充
        :param max_length: 填充的最大，为None的话补全到当前batch最大长度
        :param pad_id: 填充id，可以是单个id，也可以是字典类型，每个字典对应自己的填充id
        :param pad_mode: post 或 pre，即在后边填充和在前边填充
        :param padding_all:是否直接对所有features进行padding，这样会在迭代过程中减少padding的操作
        """
        if is_training:
            random.shuffle(features)
        self.is_training = is_training
        self.buffer_size = buffer_size

        self.features = {}
        self.columns = []
        for i in range(len(features)):
            feature = features[i]
            if isinstance(features[i], BaseClass):
                feature = features[i].dict()
            for k, v in feature.items():
                if i == 0:
                    self.features[k] = []
                    self.columns.append(k)
                self.features[k].append(v)
        self.back_columns = copy.deepcopy(self.columns)
        if not isinstance(pad_id, dict):
            pad_id = {k: pad_id for k in self.features}

        if not isinstance(max_length, dict):
            max_length = {k: max_length for k in self.features}

        self.batch_size = batch_size
        self.drop_last = drop_last

        self.idx = 0
        self.padding = padding
        self.pad_id = pad_id
        self.max_length = max_length
        self.pad_mode = pad_mode

        self.padded = False
        if padding_all:
            self.features = collate_batch(self.features, self.max_length, self.pad_id, self.pad_mode)
            self.padded = True

        self.output_types = {}
        self.output_shapes = {}

    def __num_batch__(self):
        num_features = len(self.features[self.columns[0]])
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
            return self.features[item]
        elif isinstance(item, int):
            data = {}
            for k in self.columns:
                data[k] = self.features[k][item]
        else:
            raise ValueError(f"type error of {item}")
        return data

    def __iter__(self):
        return self

    def __repr__(self):
        return f"Dataset({{\n    features: {self.columns},\n    num_batch: {self.num_batch}\n}})"

    def get_all_features(self):
        features = {}
        for k in self.columns:
            features[k] = self.features[k]
        return features

    def __next__(self):
        if self.idx < self.num_batch:
            start = self.idx * self.batch_size
            end = (self.idx + 1) * self.batch_size
            batch = {}
            for k in self.columns:
                if not self.padded and self.padding:
                    batch[k] = sequence_padding(self.features[k][start: end], self.max_length[k], self.pad_id[k],
                                                self.pad_mode)
                else:
                    batch[k] = self.features[k][start: end]
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
            if remove_column in self.features:
                self.features.pop(remove_column)
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

    def process_dataset(self, dataset: tf.data.Dataset):
        if self.is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(batch_size=self.batch_size,
                                drop_remainder=self.drop_last)
        dataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.output_types, self.output_shapes = self.get_output_types_and_shapes(dataset)
        return dataset

    def tf_gen_dataset(self):
        """
        将dataset转成tf dataset，此方法使用生成器的方式进行
        :return:
        """

        def gen():
            for i in range(len(self.features[self.columns[0]])):
                data = {}
                for k in self.columns:
                    data[k] = self.features[k][i]
                yield data

        shapes = compute_shapes(self[0], self.columns)
        types = compute_types(self[0], self.columns)
        return self.dataset_from_generator(gen, types, shapes)

    def dataset_from_generator(self, generator, types, shapes):
        dataset = tf.data.Dataset.from_generator(
            generator,
            types,
            shapes
        )
        return self.process_dataset(dataset)

    def tf_slice_dataset(self):
        """
        对应slice类型的tf dataset
        :return:
        """
        dataset = {}
        types = compute_types(self[0], self.columns)
        for k in self.columns:
            dataset[k] = tf.constant(self.features[k], dtype=types[k])
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        return self.process_dataset(dataset)

    def format_to_tf_dataset(self, dataset_type='generator'):
        assert dataset_type in ['generator', 'slice']
        if dataset_type == 'generator':
            return self.tf_gen_dataset()
        return self.tf_slice_dataset()

    @classmethod
    def get_output_types_and_shapes(cls, dataset: tf.data.Dataset, use_none=False):
        """
        根据 tf dataset，得到 dataset的 tensor shapes和 types
        :param dataset:
        :param use_none: 是否将shapes全部设置为None，这样避免bs不统一
        :return:
        """
        return compute_types_and_shapes_from_dataset(dataset, use_none)

    def output_types_and_shapes(self):
        shapes = compute_shapes(self.features, self.columns)
        types = compute_types(self[0], self.columns)
        return types, shapes
