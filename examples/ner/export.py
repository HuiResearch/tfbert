# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: export.py
@date: 2020/09/21
"""
from textToy import BertTokenizer, BertConfig, TokenClassification
from textToy.metric.ner import get_entities
from textToy.serving import load_pb, export_model_to_pb
import tensorflow.compat.v1 as tf
from time import time
import numpy as np

max_seq_length = 180
labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

# config = BertConfig.from_pretrained('ckpt/ner')
#
# input_ids = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int64, name='input_ids')
# input_mask = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int64, name='input_mask')
# token_type_ids = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int64, name='token_type_ids')
#
# model = TokenClassification(
#     model_type='bert',
#     config=config,
#     num_classes=len(labels),
#     is_training=False,
#     input_ids=input_ids,
#     input_mask=input_mask,
#     token_type_ids=token_type_ids,
#     label_ids=None,
#     add_crf=False
# )
# export_model_to_pb('ckpt/ner/model.ckpt-3936', 'pb/ner',
#                    inputs={'input_ids': input_ids, 'input_mask': input_mask, 'token_type_ids': token_type_ids},
#                    outputs={'prediction': model.predictions}
#                    )
predict_fn, input_names, output_names = load_pb('pb/ner')
tokenizer = BertTokenizer.from_pretrained('ckpt/ner', do_lower_case=True)

id2tag = dict(zip(range(len(labels)), labels))


def encode(text):
    words = list(text.strip())
    tokens = []
    for word in words:
        tokens.extend(tokenizer.tokenize(word))
    tokens = tokens[: (max_seq_length - 2)]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)

    padding_length = 180 - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    token_type_ids += [tokenizer.pad_token_type_id] * padding_length

    return input_ids, input_mask, token_type_ids


def convert_to_result(prediction, mask, text):
    tags = []
    for p, m in zip(prediction, mask):
        if m == 1:
            tags.append(id2tag[p])
        else:
            break
    tags = tags[1:-1]
    results = get_entities(tags)
    entitys = []
    for r in results:
        type_, start_pos, end_pos = r
        entitys.append((type_, text[start_pos: end_pos + 1], start_pos, end_pos))
    return entitys


while True:
    text = input(">>>")

    input_ids, input_mask, token_type_ids = encode(text)

    start = time()
    prediction = predict_fn(
        {
            'input_ids': [input_ids],
            'input_mask': [input_mask],
            'token_type_ids': [token_type_ids]
        }
    )
    end = time()
    print(">>>", convert_to_result(prediction['prediction'][0], input_mask, text))
    print("time : {:.4f}".format(end - start))