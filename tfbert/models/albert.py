# -*- coding:utf-8 -*-
# @FileName  :albert.py
# @Time      :2021/1/31 19:34
# @Author    :huanghui
import tensorflow.compat.v1 as tf
from .bert import bert_embedding as albert_embedding
from .base import BaseModel
from . import model_utils, layers, activations
from ..config import ALBertConfig


def albert_layer(input_tensor, attention_mask, config: ALBertConfig):
    with tf.variable_scope("attention_1"):
        with tf.variable_scope("self"):
            attention_layer_outputs = layers.attention(
                input_tensor,
                attention_mask=attention_mask,
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                use_relative_position=False,  # nezha使用的相对位置
                do_return_attentions_probs=config.output_attentions
            )

        attention_output = attention_layer_outputs[0]

        with tf.variable_scope("output"):
            attention_output = layers.attention_output_layer(
                input_tensor=attention_output,
                hidden_size=config.hidden_size,
                initializer_range=config.initializer_range,
                hidden_dropout_prob=config.hidden_dropout_prob
            )

    # albert 的 layer norm 所在命名空间 和 bert 不一致
    attention_output = model_utils.layer_norm(attention_output + input_tensor)
    with tf.variable_scope("ffn_1"):
        with tf.variable_scope("intermediate"):
            intermediate_output = layers.intermediate_layer(
                attention_output,
                config.intermediate_size,
                activations.get_activation(config.hidden_act),
                config.initializer_range
            )

            with tf.variable_scope("output"):
                layer_output = layers.attention_output_layer(
                    input_tensor=intermediate_output,
                    hidden_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    hidden_dropout_prob=config.hidden_dropout_prob
                )

    # albert 的 layer norm 所在命名空间 和 bert 不一致
    layer_output = model_utils.layer_norm(layer_output + attention_output)
    if config.output_attentions:
        outputs = (layer_output, attention_layer_outputs[1])
    else:
        outputs = (layer_output,)
    return outputs


def albert_encoder(input_tensor, attention_mask, config: ALBertConfig):
    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_tensor.shape[-1] != config.hidden_size:
        input_tensor = layers.dense(
            input_tensor, config.hidden_size,
            name="embedding_hidden_mapping_in",
            initializer_range=config.initializer_range
        )

    all_layer_outputs = []
    all_layer_attention_probs = []
    prev_output = input_tensor

    # albert 共享transformer参数，所以需要reuse
    with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
        for layer_idx in range(config.num_hidden_layers):
            group_idx = int(layer_idx / config.num_hidden_layers * config.num_hidden_groups)

            with tf.variable_scope("group_%d" % group_idx):
                with tf.name_scope("layer_%d" % layer_idx):
                    for inner_group_idx in range(config.inner_group_num):
                        with tf.variable_scope("inner_group_%d" % inner_group_idx):
                            layer_output = albert_layer(prev_output, attention_mask, config)
                            prev_output = layer_output[0]
                            if config.output_hidden_states:
                                all_layer_outputs.append(layer_output[0])
                            if config.output_attentions:
                                all_layer_attention_probs.append(layer_output[1])

    outputs = (prev_output,)

    if config.output_hidden_states:
        outputs = outputs + (all_layer_outputs,)

    if config.output_attentions:
        outputs = outputs + (all_layer_attention_probs,)
    return outputs  # (last layer output, all layer outputs, all layer att probs)


class ALBertModel(BaseModel):
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
            compute_type=tf.float32):
        super().__init__(config, is_training)

        input_shape = model_utils.get_shape_list(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if attention_mask is None:
            attention_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(
                scope, default_name="bert",
                reuse=tf.AUTO_REUSE if reuse else None,
                custom_getter=model_utils.get_custom_getter(compute_type)):
            with tf.variable_scope("embeddings"):
                self.embedding_output, self.embedding_table = albert_embedding(
                    config=self.config,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                )

            with tf.variable_scope("encoder"):
                attention_mask = model_utils.create_bert_mask(
                    input_ids, attention_mask)
                encoder_outputs = albert_encoder(
                    config=self.config,
                    input_tensor=tf.saturate_cast(self.embedding_output, compute_type),
                    attention_mask=attention_mask
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
