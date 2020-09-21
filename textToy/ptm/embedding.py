# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: embedding.py
@date: 2020/09/08
"""
import tensorflow.compat.v1 as tf
from .utils import (create_initializer,
                    get_shape_list)


def create_embedding(shape: list, embedding_name, initializer_range):
    return tf.get_variable(
        name=embedding_name,
        shape=shape,
        initializer=create_initializer(initializer_range))  # 初始化一个embedding table


def word_embeddings_layer(input_ids,
                          vocab_size,
                          embedding_size=128,
                          initializer_range=0.02,
                          word_embedding_name="word_embeddings",
                          use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.gather()`.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].

    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    # 创建embedding table
    embedding_table = create_embedding([vocab_size, embedding_size],
                                       word_embedding_name,
                                       initializer_range)
    # 将input_ids 铺平
    flat_input_ids = tf.reshape(input_ids, [-1])

    if use_one_hot_embeddings:
        # 通过one hot直接与embedding table相乘得到embedding
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        # gather ， 从 embedding中，根据flat_input_ids的参数值获取切片，也就是提取embedding
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)
    # 将shape还原成 bz, seq_len , emb_size
    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def token_type_embedding_layer(input_shape,
                               token_type_ids,
                               token_type_vocab_size=16,
                               token_type_embedding_name='token_type_embeddings',
                               initializer_range=0.02):
    '''
    创建token type embedding
    :param input_shape: 输入shape
    :param token_type_ids:
    :param token_type_vocab_size:
    :param token_type_embedding_name:
    :param initializer_range:
    :return:
    '''
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]
    if token_type_ids is None:
        raise ValueError("`token_type_ids` must be specified if"
                         "`use_token_type` is True.")

    token_type_table = create_embedding(
        shape=[token_type_vocab_size, width],
        embedding_name=token_type_embedding_name,
        initializer_range=initializer_range
    )
    # 铺平输入ids
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    return token_type_embeddings


def position_embeddings_layer(input_shape,
                              position_embedding_name="position_embeddings",
                              initializer_range=0.02,
                              max_position_embeddings=512
                              ):
    seq_length = input_shape[1]
    width = input_shape[2]

    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)

    # control_dependencies是 tensorflow 中的一个flow顺序控制机制
    # 在此处, 运行一下代码块之前会先运行assert op，主要检查输入长度是否小于支持的最大长度
    with tf.control_dependencies([assert_op]):
        full_position_embeddings = create_embedding(
            shape=[max_position_embeddings, width],
            embedding_name=position_embedding_name,
            initializer_range=initializer_range
        )
        # 直接使用切片取embedding
        position_embeddings = tf.slice(full_position_embeddings,
                                       [0, 0],
                                       [seq_length, -1]
                                       )
        num_dims = len(input_shape)
        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        position_broadcast_shape.extend([seq_length, width])
        position_embeddings = tf.reshape(position_embeddings,
                                         position_broadcast_shape)

    return position_embeddings
