# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: classification.py
@date: 2020/09/09
"""
from functools import partial
import tensorflow.compat.v1 as tf
from . import process_dataset, BaseClass, multiple_convert_examples_to_features, single_example_to_features


def return_types_and_shapes(for_trainer, is_multi_label=False):
    if for_trainer:
        shape = tf.TensorShape([None, None])
        label_shape = tf.TensorShape([None])
    else:
        shape = tf.TensorShape([None])
        label_shape = tf.TensorShape([])

    output_types = {"input_ids": tf.int32,
                    "input_mask": tf.int32,
                    "token_type_ids": tf.int32,
                    'label_ids': tf.int32}
    output_shapes = {"input_ids": shape,
                     "input_mask": shape,
                     "token_type_ids": shape,
                     'label_ids': label_shape}
    if is_multi_label:
        output_shapes['label_ids'] = shape
    return output_types, output_shapes


class InputExample(BaseClass):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(BaseClass):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 label_id=None,
                 ex_id=None):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.ex_id = ex_id


def convert_example_to_feature(example: InputExample,
                               max_length=512,
                               label_map=None,
                               is_multi_label=False) -> InputFeature:
    inputs = tokenizer.encode(
        example.text_a,  # 传入句子 a
        text_pair=example.text_b,  # 传入句子 b，可以为None
        add_special_tokens=True,  # 是否增加 cls  sep
        max_length=max_length,  # 最大长度
        pad_to_max_length=True  # 是否将句子padding到最大长度
    )
    if example.label is not None:
        # 多标签分类的话，先将label设为one hot 类型
        if is_multi_label:
            label_id = [0] * len(label_map)
            for lb in example.label:
                label_id[label_map[lb]] = 1
        else:
            label_id = label_map[example.label]
    else:
        label_id = None
    return InputFeature(
        guid=0,
        input_ids=inputs['input_ids'],
        input_mask=inputs['input_mask'],
        token_type_ids=inputs['token_type_ids'],
        label_id=label_id,
        ex_id=example.guid
    )


def convert_example_to_feature_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        label_list=None,
        set_type='train',
        is_multi_label=False,
        threads=1
):
    '''
    将examples转为features, 适用于单句和双句分类任务
    :param examples:
    :param tokenizer: bert分词器
    :param max_length: 句子最大长度
    :param label_list: 标签
    :param set_type:
    :param is_multi_label: 是否是多标签分类
    :param threads:
    :return:
    '''

    label_map = None
    if label_list is not None:
        label_map = {label: i for i, label in enumerate(label_list)}
    annotate_ = partial(
        convert_example_to_feature,
        max_length=max_length,
        label_map=label_map,
        is_multi_label=is_multi_label
    )
    if threads > 1:
        features = multiple_convert_examples_to_features(
            examples,
            annotate_=annotate_,
            initializer=convert_example_to_feature_init,
            initargs=(tokenizer,),
            threads=threads
        )
    else:
        convert_example_to_feature_init(tokenizer)
        features = single_example_to_features(
            examples, annotate_=annotate_
        )
    new_features = []
    i = 0
    for feature in features:
        feature.guid = set_type + '-' + str(i)
        new_features.append(feature)
    return new_features


def create_dataset_by_gen(
        features, batch_size,
        set_type='train',
        is_multi_label=False
):
    '''
    通过生成器的方式包装dataset
    :param features:
    :param batch_size:
    :param set_type:
    :param is_multi_label: 是否是多标签任务
    :return:
    '''

    def gen():
        for ex in features:
            yield {
                "input_ids": ex.input_ids,
                "input_mask": ex.input_mask,
                "token_type_ids": ex.token_type_ids,
                'label_ids': ex.label_id,
            }

    output_types, output_shapes = return_types_and_shapes(
        for_trainer=False, is_multi_label=is_multi_label)

    # 这种方式需要传入生成器，定义好数据类型，数据的shape
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types,
        output_shapes
    )

    return process_dataset(dataset, batch_size, len(features), set_type)


def create_dataset_from_slices(
        features, batch_size,
        set_type='train',
        is_multi_label=False
):
    '''
    通过生成器的方式包装dataset
    :param features:
    :param batch_size:
    :param set_type:
    :param is_multi_label: 是否是多标签任务
    :return:
    '''
    dataset = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                [f.input_ids for f in features],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                [f.input_mask for f in features],
                dtype=tf.int32),
        "token_type_ids":
            tf.constant(
                [f.token_type_ids for f in features],
                dtype=tf.int32),
        "label_ids":
            tf.constant([f.label_id for f in features], dtype=tf.int32),
    })
    return process_dataset(dataset, batch_size, len(features), set_type)