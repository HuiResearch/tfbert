# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: ner.py
@date: 2020/09/12
"""
from functools import partial
from . import BaseClass, multiple_convert_examples_to_features, single_example_to_features
from ..tokenizer import GlyceBertTokenizer
from typing import List


class InputExample(BaseClass):
    def __init__(self, guid, words: List[str], tags: List[str] = None):
        self.guid = guid
        self.words = words
        self.tags = tags


class InputFeature(BaseClass):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids: List[int],
                 attention_mask: List[int] = None,
                 token_type_ids: List[int] = None,
                 pinyin_ids=None,
                 label_ids: List[int] = None,
                 tok_to_orig_index: List[int] = None,
                 ex_id=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.pinyin_ids = pinyin_ids
        self.label_ids = label_ids
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

    attention_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)

    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    token_type_ids += [tokenizer.pad_token_type_id] * padding_length
    if has_label:
        label_ids += [pad_token_label_id] * padding_length

    if isinstance(tokenizer, GlyceBertTokenizer):
        pinyin_ids = tokenizer.convert_token_ids_to_pinyin_ids(input_ids)
    else:
        pinyin_ids = None

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(token_type_ids) == max_length
    if has_label:
        assert len(label_ids) == max_length

    return InputFeature(
        guid=str(0),
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        pinyin_ids=pinyin_ids,
        label_ids=label_ids if has_label else None,
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
        threads=1
) -> List[InputFeature]:
    label_map = None
    if label_list is not None:
        label_map = {label: i for i, label in enumerate(label_list)}
    annotate_ = partial(
        convert_example_to_feature,
        max_length=max_length,
        label_map=label_map,
        pad_token_label_id=pad_token_label_id
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
