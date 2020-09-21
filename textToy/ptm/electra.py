# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: electra.py
@date: 2020/09/08
"""
from .bert import BertEmbedding as ELECTRAEmbedding
from .bert import BertEncoder as ELECTRAEncoder
from .utils import (
    get_shape_list,
    create_attention_mask_from_input_mask)
import tensorflow.compat.v1 as tf
from .base import BaseModel


class ELECTRAModel(BaseModel):
    def __init__(
            self,
            config,
            input_ids,
            input_mask=None,
            token_type_ids=None,
            is_training=None,
            scope=None,
            reuse=False
    ):
        super().__init__(config, is_training)

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="electra",
                               reuse=tf.AUTO_REUSE if reuse else None):
            with tf.variable_scope("embeddings"):
                self.embedding_output, self.embedding_table = ELECTRAEmbedding(
                    config=self.config,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )

            with tf.variable_scope("encoder"):
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)
                encoder_outputs = ELECTRAEncoder(
                    config=self.config,
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask
                )

        # electra 的 pool output是直接返回first token的vec
        pooled_output = encoder_outputs[0][:, 0]
        # (pooled output, sequence output, all layer outputs, all layer att probs)
        self.outputs = (pooled_output,) + encoder_outputs
