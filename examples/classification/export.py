# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: export.py
@date: 2020/09/21
"""
from textToy import BertTokenizer, SequenceClassification, BertConfig
from textToy.serving import load_pb, export_model_to_pb
from run_ptm import labels
import tensorflow.compat.v1 as tf
from time import time
import numpy as np


# config = BertConfig.from_pretrained('ckpt/classification')
# input_ids = tf.placeholder(shape=[None, 32], dtype=tf.int64, name='input_ids')
# input_mask = tf.placeholder(shape=[None, 32], dtype=tf.int64, name='input_mask')
# token_type_ids = tf.placeholder(shape=[None, 32], dtype=tf.int64, name='token_type_ids')
# model = SequenceClassification(
#     'bert',
#     config,
#     5,
#     False,
#     input_ids,
#     input_mask=input_mask,
#     token_type_ids=token_type_ids,
#     label_ids=None,
#     dropout_prob=0.1
# )
# export_model_to_pb('ckpt/classification', 'pb/classification',
#                    inputs={'input_ids': input_ids, 'input_mask': input_mask, 'token_type_ids': token_type_ids},
#                    outputs={'logits': model.logits})
predict_fn, input_names, output_names = load_pb('pb/classification')
tokenizer = BertTokenizer.from_pretrained('ckpt/classification', do_lower_case=True)
while True:
    text = input(">>>")

    inputs = tokenizer.encode(text,
                              add_special_tokens=True, max_length=32,
                              pad_to_max_length=True)
    start = time()
    prediction = predict_fn(
        {
            'input_ids': [inputs['input_ids']],
            'input_mask': [inputs['input_mask']],
            'token_type_ids': [inputs['token_type_ids']]
        }
    )
    end = time()
    print(">>>", labels[np.argmax(prediction['logits'], -1)[0]])
    print("time : {:.4f}".format(end - start))