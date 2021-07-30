# -*- coding:utf-8 -*-
# @FileName  :glyce_bert.py
# @Time      :2021/7/29 14:11
# @Author    :huanghui
import os
import json
import tensorflow.compat.v1 as tf
from . import embeddings, layers, model_utils
from .base import BaseModel
from .bert import bert_encoder


def glyph_bert_embeddings(
        config,
        input_ids,
        pinyin_ids,
        token_type_ids=None
):
    (word_embeddings, embedding_table) = embeddings.create_word_embeddings(
        input_ids=input_ids,
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        initializer_range=config.initializer_range,
        word_embedding_name="word_embeddings"
    )

    with open(os.path.join(config.config_path, 'pinyin_map.json')) as fin:
        pinyin_dict = json.load(fin)
    pinyin_embeddings = embeddings.create_pinyin_embeddings(
        pinyin_ids,
        embedding_size=128,
        pinyin_out_dim=config.embedding_size,
        initializer_range=config.initializer_range,
        pinyin_vocab_size=len(pinyin_dict['idx2char']))

    font_files = []
    for file in os.listdir(config.config_path):
        if file.endswith(".npy"):
            font_files.append(os.path.join(config.config_path, file))
    glyph_embeddings = embeddings.create_glyph_embeddings(
        input_ids, font_files
    )
    glyph_embeddings = layers.dense(glyph_embeddings, config.embedding_size, name="glyph_map")

    # fusion layer
    concat_embeddings = tf.concat([word_embeddings, pinyin_embeddings, glyph_embeddings], axis=2)
    inputs_embeds = layers.dense(concat_embeddings, config.embedding_size, name='map_fc')

    token_type_embeddings = embeddings.create_token_type_embeddings(
        token_type_ids=token_type_ids,
        embedding_size=config.embedding_size,
        token_type_vocab_size=config.type_vocab_size,
        token_type_embedding_name='token_type_embeddings',
        initializer_range=config.initializer_range
    )

    position_embeddings = embeddings.create_position_embeddings(
        seq_len=model_utils.get_shape_list(input_ids)[1],
        embedding_size=config.embedding_size,
        position_embedding_name='position_embeddings',
        initializer_range=config.initializer_range,
        max_position_embeddings=config.max_position_embeddings
    )

    embedding_output = inputs_embeds + position_embeddings + token_type_embeddings
    embedding_output = model_utils.layer_norm_and_dropout(
        embedding_output,
        config.hidden_dropout_prob
    )

    return embedding_output, embedding_table


class GlyceBertModel(BaseModel):
    def __init__(
            self,
            config,
            is_training,
            input_ids,
            pinyin_ids,
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
                self.embedding_output, self.embedding_table = glyph_bert_embeddings(
                    config=self.config,
                    input_ids=input_ids,
                    pinyin_ids=pinyin_ids,
                    token_type_ids=token_type_ids
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
