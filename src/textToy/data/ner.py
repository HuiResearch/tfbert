# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: ner.py
@date: 2020/09/12
"""
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tensorflow.compat.v1 as tf
from . import process_dataset
from typing import List


def return_types_and_shapes(for_trainer):
    if for_trainer:
        shape = tf.TensorShape([None, None])
        label_shape = tf.TensorShape([None, None])
    else:
        shape = tf.TensorShape([None])
        label_shape = tf.TensorShape([None])

    output_types = {"input_ids": tf.int32,
                    "input_mask": tf.int32,
                    "token_type_ids": tf.int32,
                    'label_ids': tf.int64}
    output_shapes = {"input_ids": shape,
                     "input_mask": shape,
                     "token_type_ids": shape,
                     'label_ids': label_shape}

    return output_types, output_shapes


class InputExample:
    def __init__(self, guid, words: List[str], tags: List[str] = None):
        self.guid = guid
        self.words = words
        self.tags = tags


class InputFeature(object):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids: List[int],
                 input_mask: List[int] = None,
                 token_type_ids: List[int] = None,
                 label_id: List[int] = None,
                 tok_to_orig_index: List[int] = None,
                 ex_id=None):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.ex_id = ex_id
        self.tok_to_orig_index = tok_to_orig_index


def convert_example_to_feature(
        example: InputExample,
        max_length=512,
        label_map=None,
        pad_token_label_id=0
):
    has_label = bool(example.tags is not None)
    tokens = []
    label_ids = []
    tok_to_orig_index = []  # 用来存放token 和 原始words列表的位置对应关系，因为bert分词可能会将一个word分成多个token
    for i in range(len(example.words)):
        word = example.words[i]
        if has_label:
            label = example.tags[i]
        word_tokens = tokenizer.tokenize(word)

        if len(word_tokens) > 0:
            tok_to_orig_index.append(i)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if has_label:
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

    special_tokens_count = tokenizer.num_special_tokens
    if len(tokens) > max_length - special_tokens_count:
        tokens = tokens[: (max_length - special_tokens_count)]
        label_ids = label_ids[: (max_length - special_tokens_count)]

    tokens += [tokenizer.sep_token]

    if has_label:
        label_ids += [pad_token_label_id]

    token_type_ids = [0] * len(tokens)

    tokens = [tokenizer.cls_token] + tokens
    if has_label:
        label_ids = [pad_token_label_id] + label_ids
    token_type_ids = [0] + token_type_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)

    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    token_type_ids += [tokenizer.pad_token_type_id] * padding_length
    if has_label:
        label_ids += [pad_token_label_id] * padding_length

    assert len(input_ids) == max_length
    assert len(input_mask) == max_length
    assert len(token_type_ids) == max_length
    if has_label:
        assert len(label_ids) == max_length

    return InputFeature(
        guid=str(0),
        input_ids=input_ids, input_mask=input_mask,
        token_type_ids=token_type_ids,
        label_id=label_ids if has_label else None,
        tok_to_orig_index=tok_to_orig_index
    )


def convert_example_to_feature_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(
        examples: List[InputExample],
        tokenizer,
        max_length=512,
        label_list=None,
        set_type='train',
        pad_token_label_id=0,
        use_multi_threads=False,
        threads=1
) -> List[InputFeature]:
    label_map = None
    if label_list is not None:
        label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    threads = min(threads, cpu_count())
    if use_multi_threads and threads > 1:
        with Pool(threads, initializer=convert_example_to_feature_init, initargs=(tokenizer,)) as p:
            annotate_ = partial(
                convert_example_to_feature,
                max_length=max_length,
                label_map=label_map,
                pad_token_label_id=pad_token_label_id
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert examples to features"
                )
            )
    else:
        convert_example_to_feature_init(tokenizer)
        for example in tqdm(examples, desc="convert examples to features"):
            features.append(convert_example_to_feature(
                example,
                max_length=max_length,
                label_map=label_map,
                pad_token_label_id=pad_token_label_id
            ))
    new_features = []
    i = 0
    for feature in features:
        feature.guid = set_type + '-' + str(i)
        new_features.append(feature)
    return new_features


def create_dataset_by_gen(
        features, batch_size,
        set_type='train'
):
    '''
    通过生成器的方式包装dataset
    :param features:
    :param batch_size:
    :param set_type:
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
        for_trainer=False)

    # 这种方式需要传入生成器，定义好数据类型，数据的shape
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types,
        output_shapes
    )
    return process_dataset(dataset, batch_size, len(features), set_type)
