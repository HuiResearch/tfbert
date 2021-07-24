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

    flat_input_ids = tf.reshape(input_ids, [-1])
    output = tf.gather(embedding_table, flat_input_ids)
    input_shape = model_utils.get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape + [embedding_size])
    # output = tf.nn.embedding_lookup(embedding_table, input_ids)

    return (output, embedding_table)


def create_token_type_embeddings(
        token_type_ids,
        embedding_size,
        token_type_vocab_size=2,
        token_type_embedding_name='token_type_embeddings',
        initializer_range=0.02):
    input_shape = model_utils.get_shape_list(token_type_ids)
    token_type_table = model_utils.create_weight(
        shape=[token_type_vocab_size, embedding_size],
        var_name=token_type_embedding_name,
        initializer_range=initializer_range
    )
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [input_shape[0], input_shape[1], -1])
    # token_type_embeddings = tf.nn.embedding_lookup(token_type_table, token_type_ids)
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
    position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                   [seq_len, -1])
    # num_dims = len(output.shape.as_list())

    # Only the last two dimensions are relevant (`seq_length` and `width`), so
    # we broadcast among the first dimensions, which is typically just
    # the batch size.
    position_broadcast_shape = []
    for _ in range(1):
        position_broadcast_shape.append(1)
    position_broadcast_shape.extend([seq_len, embedding_size])
    position_embeddings = tf.reshape(position_embeddings,
                                     position_broadcast_shape)
    # position_embeddings = tf.nn.embedding_lookup(full_position_embeddings, tf.range(0, seq_len))

    return position_embeddings
