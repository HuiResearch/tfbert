# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: __init__.py.py
@date: 2020/09/08
"""


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
