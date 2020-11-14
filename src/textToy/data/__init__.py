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


class BaseClass:
    def dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def __str__(self):
        return "{} \n {}".format(
            self.__class__.__name__, json.dumps(self.dict(), ensure_ascii=False))


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


def process_dataset(dataset, batch_size, num_features, set_type):
    if set_type == 'train':
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=num_features)
    dataset = dataset.batch(batch_size=batch_size,
                            drop_remainder=bool(set_type == 'train'))
    dataset.prefetch(batch_size)

    # 在train阶段，因为设置了drop_remainder，会舍弃不足batch size的一个batch，所以步数计算方式和验证测试不同
    if set_type == 'train':
        num_batch_per_epoch = num_features // batch_size
    else:
        num_batch_per_epoch = (num_features + batch_size - 1) // batch_size
    return dataset, num_batch_per_epoch
