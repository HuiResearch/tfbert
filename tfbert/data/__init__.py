# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: __init__.py.py
@date: 2020/09/08
"""
import json
import copy
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import numpy as np
import tensorflow.compat.v1 as tf


class BaseClass:
    def dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def __str__(self):
        return "{} \n {}".format(
            self.__class__.__name__, json.dumps(self.dict(), ensure_ascii=False))

    def keys(self):
        return list(self.dict().keys())

    def __getitem__(self, item):
        return self.dict()[item]

    def __contains__(self, item):
        return item in self.dict()


def single_example_to_features(
        examples, annotate_, desc='convert examples to feature'):
    features = []
    for example in tqdm(examples, desc=desc):
        features.append(annotate_(example))
    return features


def multiple_convert_examples_to_features(
        examples,
        annotate_,
        initializer,
        initargs,
        threads,
        desc='convert examples to feature'):
    threads = min(cpu_count(), threads)
    features = []
    with Pool(threads, initializer=initializer, initargs=initargs) as p:
        features = list(tqdm(
            p.imap(annotate_, examples, chunksize=32),
            total=len(examples),
            desc=desc
        ))
    return features


def process_dataset(dataset, batch_size, num_features, set_type, buffer_size=None):
    if set_type == 'train':
        if buffer_size is None:
            buffer_size = num_features
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size=batch_size,
                            drop_remainder=bool(set_type == 'train'))
    dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # 在train阶段，因为设置了drop_remainder，会舍弃不足batch size的一个batch，所以步数计算方式和验证测试不同
    if set_type == 'train':
        num_batch_per_epoch = num_features // batch_size
    else:
        num_batch_per_epoch = (num_features + batch_size - 1) // batch_size
    return dataset, num_batch_per_epoch


def compute_types(example, columns=None):
    if columns is None:
        columns = list(example.keys())

    def fn(values):
        if isinstance(values, np.ndarray):
            if values.dtype == np.dtype(float):
                return tf.float32
            elif values.dtype == np.int64:
                return tf.int32  # 统一使用int32
            elif values.dtype == np.int32:
                return tf.int32
            else:
                raise ValueError(
                    f"values={values} is an np.ndarray with items of dtype {values.dtype}, which cannot be supported"
                )
        # 支持到二维矩阵。。。
        elif (isinstance(values, list) and isinstance(values[0], float)) or isinstance(values, float):
            return tf.float32
        elif (isinstance(values, list) and isinstance(values[0], int)) or isinstance(values, int):
            return tf.int32
        elif (isinstance(values, list) and isinstance(values[0], str)) or isinstance(values, str):
            return tf.string
        elif isinstance(values, list) and isinstance(values[0], list):
            return fn(values[0])
        else:
            raise ValueError(f"values={values} has dtype {values.dtype}, which cannot be supported")

    tf_types = {}
    for k in columns:
        if k in example:
            tf_types[k] = fn(example[k])
    return tf_types


def compute_shapes(example, columns=None):
    if columns is None:
        columns = list(example.keys())

    def fn(array):
        np_shape = np.shape(array)
        return [None] * len(np_shape)

    tf_shapes = {}
    for k in columns:
        if k in example:
            tf_shapes[k] = fn(example[k])
    return tf_shapes


def compute_types_and_shapes_from_dataset(dataset: tf.data.Dataset, use_none=False):
    """
    根据 tf dataset，得到dataset的tensor shapes和types
    :param dataset:
    :param use_none: 是否将shapes全部设置为None，这样避免bs不统一
    :return:
    """

    def to_none(tensor_shape: tf.TensorShape):
        return tf.TensorShape([None] * len(tensor_shape.as_list()))

    output_types = tf.data.get_output_types(dataset)
    output_shapes = tf.data.get_output_shapes(dataset)
    if use_none:
        for k in output_shapes:
            output_shapes[k] = to_none(output_shapes[k])
    return output_types, output_shapes


from .dataset import Dataset, collate_batch, sequence_padding
