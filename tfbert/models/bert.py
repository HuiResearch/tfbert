# -*- coding:utf-8 -*-
# @FileName  :bert.py
# @Time      :2021/1/31 17:19
# @Author    :huanghui
import tensorflow.compat.v1 as tf
from . import embeddings, layers, model_utils, activations
from .base import BaseModel


def bert_embedding(
        config,
        input_ids,
        token_type_ids=None,
        add_position_embedding=True
):
    (embedding_output, embedding_table) = embeddings.create_word_embeddings(
        input_ids=input_ids,
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        initializer_range=config.initializer_range,
        word_embedding_name="word_embeddings"
    )

    token_type_embeddings = embeddings.create_token_type_embeddings(
        token_type_ids=token_type_ids,
        embedding_size=config.embedding_size,
        token_type_vocab_size=config.type_vocab_size,
        token_type_embedding_name='token_type_embeddings',
        initializer_range=config.initializer_range
    )
    embedding_output += token_type_embeddings

    if add_position_embedding:
        position_embeddings = embeddings.create_position_embeddings(
            seq_len=model_utils.get_shape_list(input_ids)[1],
            embedding_size=config.embedding_size,
            position_embedding_name='position_embeddings',
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings
        )
        embedding_output += position_embeddings

    embedding_output = model_utils.layer_norm_and_dropout(
        embedding_output,
        config.hidden_dropout_prob
    )
    return embedding_output, embedding_table


def bert_layer(input_tensor, attention_mask, config, use_relative_position=False):
    with tf.variable_scope("attention"):
        # 多头自注意力层
        with tf.variable_scope("self"):
            attention_layer_outputs = layers.attention(
                input_tensor,
                attention_mask=attention_mask,
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                use_relative_position=use_relative_position,  # nezha使用的相对位置
                do_return_attentions_probs=config.output_attentions
            )
            attention_output = attention_layer_outputs[0]

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        # attention输出层
        with tf.variable_scope("output"):
            attention_output = layers.attention_output_layer(
                input_tensor=attention_output,
                hidden_size=config.hidden_size,
                initializer_range=config.initializer_range,
                hidden_dropout_prob=config.hidden_dropout_prob
            )
            attention_output = model_utils.layer_norm(attention_output + input_tensor)

    # transformer中间层
    # The activation is only applied to the "intermediate" hidden layer.
    with tf.variable_scope("intermediate"):
        intermediate_output = layers.intermediate_layer(
            attention_output,
            config.intermediate_size,
            activations.get_activation(config.hidden_act),
            config.initializer_range
        )

    # Down-project back to `hidden_size` then add the residual.
    # transformer 输出层
    with tf.variable_scope("output"):
        layer_output = layers.attention_output_layer(
            input_tensor=intermediate_output,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
            hidden_dropout_prob=config.hidden_dropout_prob
        )
        layer_output = model_utils.layer_norm(layer_output + attention_output)
    if config.output_attentions:
        outputs = (layer_output, attention_layer_outputs[1])
    else:
        outputs = (layer_output,)
    return outputs


def bert_encoder(input_tensor, attention_mask,
                 config, use_relative_position=False):
    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.

    all_layer_outputs = []
    all_layer_attention_probs = []
    prev_output = input_tensor
    for layer_idx in range(config.num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):

            layer_output = bert_layer(
                prev_output, attention_mask, config,
                use_relative_position=use_relative_position
            )
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


class BertModel(BaseModel):
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
                    add_position_embedding=True
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
                    use_relative_position=False
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
