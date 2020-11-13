# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: albert.py
@date: 2020/09/08
"""
from .bert import BertEmbedding as ALBertEmbedding
from .transformer import (
    albert_transformer_layer,
    pooler_layer,
    dense_layer_2d)

from .utils import (
    reshape_to_matrix,
    reshape_from_matrix,
    get_shape_list,
    create_attention_mask_from_input_mask)

from .utils import get_activation
import tensorflow.compat.v1 as tf
from .base import BaseModel


def ALBertEncoder(
        config,
        input_tensor,
        attention_mask=None
):
    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (config.hidden_size, config.num_attention_heads))

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    all_layer_outputs = []
    all_layer_attention_probs = []

    # ALBert 使用了因式矩阵分解，将embedding size 调小，从而减少参数
    # 因此，在embedding 和 transformer之间，增加了一个变换矩阵，
    # 将embedding 从embedding size 映射到 hidden size
    if input_width != config.hidden_size:
        prev_output = dense_layer_2d(
            input_tensor, config.hidden_size, config.initializer_range,
            None, use_einsum=True, name="embedding_hidden_mapping_in"
        )
    else:
        prev_output = input_tensor

    prev_output = reshape_to_matrix(prev_output)

    # albert 共享transformer参数，所以需要reuse
    with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
        for layer_idx in range(config.num_hidden_layers):
            group_idx = int(layer_idx / config.num_hidden_layers * config.num_hidden_groups)

            with tf.variable_scope("group_%d" % group_idx):
                with tf.name_scope("layer_%d" % layer_idx):
                    layer_output = prev_output
                    for inner_group_idx in range(config.inner_group_num):
                        with tf.variable_scope("inner_group_%d" % inner_group_idx):
                            layer_output = albert_transformer_layer(
                                layer_input=layer_output,
                                batch_size=batch_size,
                                seq_length=seq_length,
                                attention_mask=attention_mask,
                                hidden_size=config.hidden_size,  # 隐藏层大小
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                intermediate_act_fn=get_activation(config.hidden_act),
                                hidden_dropout_prob=config.hidden_dropout_prob,
                                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                initializer_range=config.initializer_range,
                                do_return_attentions_probs=config.output_attentions
                            )
                            prev_output = layer_output[0]
                            if config.output_hidden_states:
                                all_layer_outputs.append(layer_output[0])
                            if config.output_attentions:
                                all_layer_attention_probs.append(layer_output[1])

    final_output = reshape_from_matrix(prev_output, input_shape)

    outputs = (final_output,)

    if config.output_hidden_states:
        final_all_layer_outputs = []
        for layer_output in all_layer_outputs:
            # 在transformer layer中，所有输出都是转为2D矩阵进行dense运算
            # 所以需要把输出还原成 [bz, seq_len, hz]
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_all_layer_outputs.append(final_output)
        outputs = outputs + (final_all_layer_outputs,)

    if config.output_attentions:
        outputs = outputs + (all_layer_attention_probs,)
    return outputs  # (last layer output, all layer outputs, all layer att probs)


class ALBertModel(BaseModel):
    def __init__(
            self,
            config,
            input_ids,
            input_mask=None,
            token_type_ids=None,
            is_training=None,
            scope=None,
            reuse=False):
        super().__init__(config, is_training)

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert",
                               reuse=tf.AUTO_REUSE if reuse else None
                               ):
            with tf.variable_scope("embeddings"):
                self.embedding_output, self.embedding_table = ALBertEmbedding(
                    config=self.config,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                )

            with tf.variable_scope("encoder"):
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)
                encoder_outputs = ALBertEncoder(
                    config=self.config,
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask
                )

            with tf.variable_scope("pooler"):
                pooled_output = pooler_layer(
                    sequence_output=encoder_outputs[0],
                    hidden_size=self.config.hidden_size,
                    initializer_range=self.config.initializer_range
                )
        # (pooled output, sequence output, all layer outputs, all layer att probs)
        self.outputs = (pooled_output,) + encoder_outputs
