# -*- coding:utf-8 -*-
# @FileName  :nezha.py
# @Time      :2021/2/2 20:54
# @Author    :huanghui
import tensorflow.compat.v1 as tf
from . import layers, model_utils
from .base import BaseModel
from .bert import bert_embedding, bert_encoder


class NezhaModel(BaseModel):
    def __init__(
            self,
            config,
            is_training,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            return_pool=True,
            scope=None,
            reuse=False,
            compute_type=tf.float32
    ):
        super().__init__(config, is_training)

        input_shape = model_utils.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if attention_mask is None:
            attention_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int64)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int64)

        with tf.variable_scope(
                scope, default_name="bert",
                reuse=tf.AUTO_REUSE if reuse else None,
                custom_getter=model_utils.get_custom_getter(compute_type)):
            with tf.variable_scope("embeddings"):
                self.embedding_output, self.embedding_table = bert_embedding(
                    config=self.config,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    add_position_embedding=False
                )

            with tf.variable_scope("encoder"):
                attention_mask = model_utils.create_bert_mask(
                    input_ids, attention_mask)
                if model_utils.get_shape_list(self.embedding_output)[-1] != self.config.hidden_size:
                    self.embedding_output = layers.dense(
                        self.embedding_output, self.config.hidden_size,
                        'embedding_hidden_mapping_in', initializer_range=self.config.initializer_range
                    )
                encoder_outputs = bert_encoder(
                    input_tensor=tf.saturate_cast(self.embedding_output, compute_type),
                    attention_mask=attention_mask,
                    config=self.config,
                    use_relative_position=True
                )
            if return_pool:
                with tf.variable_scope("pooler"):
                    pooled_output = layers.pooler_layer(
                        sequence_output=encoder_outputs[0],
                        hidden_size=self.config.hidden_size,
                        initializer_range=self.config.initializer_range
                    )
            else:
                pooled_output = None
        # (pooled output, sequence output, all layer outputs, all layer att probs)
        self.outputs = (pooled_output,) + encoder_outputs
