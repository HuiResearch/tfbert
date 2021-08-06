# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: classification.py
@date: 2020/09/09
"""
from functools import partial
from . import BaseClass, multiple_convert_examples_to_features, single_example_to_features


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
                 attention_mask=None,
                 token_type_ids=None,
                 pinyin_ids=None,
                 label_ids=None,
                 ex_id=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.pinyin_ids = pinyin_ids
        self.label_ids = label_ids
        self.ex_id = ex_id


def convert_example_to_feature(example: InputExample,
                               max_length=512,
                               label_map=None,
                               is_multi_label=False) -> InputFeature:
    """
    text,
               text_pair=None,
               max_length=512,
               pad_to_max_len=False,
               truncation_strategy="longest_first",
               return_position_ids=False,
               return_token_type_ids=True,
               return_attention_mask=True,
               return_length=False,
               return_overflowing_tokens=False,
               return_special_tokens_mask=False
    :param example:
    :param max_length:
    :param label_map:
    :param is_multi_label:
    :return:
    """
    inputs = tokenizer(
        example.text_a,  # 传入句子 a
        text_pair=example.text_b,  # 传入句子 b，可以为None
        max_length=max_length,  # 最大长度
        padding="max_length",  # 是否将句子padding到最大长度
        truncation=True
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
        attention_mask=inputs['attention_mask'],
        token_type_ids=inputs['token_type_ids'],
        pinyin_ids=inputs['pinyin_ids'] if "pinyin_ids" in inputs else None,
        label_ids=label_id,
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
