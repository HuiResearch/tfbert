# -*- coding:utf-8 -*-
# @FileName  :embeddings.py
# @Time      :2021/1/31 15:32
# @Author    :huanghui
import tensorflow.compat.v1 as tf
from . import model_utils


def create_word_embeddings(
        input_ids,
        vocab_size,
        embedding_size=128,
        initializer_range=0.02,
        word_embedding_name="word_embeddings"):
    # 创建embedding table
    embedding_table = model_utils.create_weight(
        [vocab_size, embedding_size],
        word_embedding_name,
        initializer_range)

    output = tf.nn.embedding_lookup(embedding_table, input_ids)

    return (output, embedding_table)


def create_token_type_embeddings(
        token_type_ids,
        embedding_size,
        token_type_vocab_size=2,
        token_type_embedding_name='token_type_embeddings',
        initializer_range=0.02):
    token_type_table = model_utils.create_weight(
        shape=[token_type_vocab_size, embedding_size],
        var_name=token_type_embedding_name,
        initializer_range=initializer_range
    )

    token_type_embeddings = tf.nn.embedding_lookup(token_type_table, token_type_ids)
    return token_type_embeddings


def create_position_embeddings(
        seq_len,
        embedding_size,
        position_embedding_name="position_embeddings",
        initializer_range=0.02,
        max_position_embeddings=512
):

    full_position_embeddings = model_utils.create_weight(
        shape=[max_position_embeddings, embedding_size],
        var_name=position_embedding_name,
        initializer_range=initializer_range
    )
    position_embeddings = tf.nn.embedding_lookup(full_position_embeddings, tf.range(0, seq_len))

    return position_embeddings
