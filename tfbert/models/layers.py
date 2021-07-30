# -*- coding:utf-8 -*-
# @FileName  :layers.py
# @Time      :2021/1/31 15:24
# @Author    :huanghui

import tensorflow.compat.v1 as tf
from tensorflow.python.ops import gen_nn_ops
import math
from . import model_utils, activations
from . import crf
from typing import List


def dense(input_tensor, output_size, name, activation=None, initializer_range=0.02):
    """
    本来以为dense只支持2d的，但发现三维也支持。。。白改了
    :param input_tensor:
    :param output_size:
    :param name:
    :param activation:
    :param initializer_range:
    :return:
    """
    output = tf.layers.dense(
        input_tensor,
        output_size,
        name=name,
        activation=activations.get_activation(activation),
        kernel_initializer=model_utils.create_initializer(initializer_range))
    return output


def attention(
        input_tensor,
        attention_mask=None,
        hidden_size=768,
        num_attention_heads=12,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        use_relative_position=False,  # nezha使用的相对位置
        do_return_attentions_probs=False):
    """
    self attention层
    :param input_tensor: bz， seq_len , hidden_size
    :param attention_mask:
    :param hidden_size:
    :param num_attention_heads:
    :param attention_probs_dropout_prob:
    :param initializer_range:
    :param use_relative_position:
    :param do_return_attentions_probs:
    :return:
    """
    def transpose(tensor, shape):
        output_tensor = tf.reshape(tensor, shape=shape)
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    shape = model_utils.get_shape_list(input_tensor, expected_rank=3)

    size_per_head = hidden_size // num_attention_heads
    query_layer = dense(
        input_tensor=input_tensor,
        output_size=num_attention_heads * size_per_head,
        activation=None,
        name='query',
        initializer_range=initializer_range
    )
    key_layer = dense(
        input_tensor=input_tensor,
        output_size=num_attention_heads * size_per_head,
        activation=None,
        name='key',
        initializer_range=initializer_range
    )
    value_layer = dense(
        input_tensor=input_tensor,
        output_size=num_attention_heads * size_per_head,
        activation=None,
        name='value',
        initializer_range=initializer_range
    )

    query_layer = transpose(
        query_layer, shape=shape[:-1] + [num_attention_heads, size_per_head])

    key_layer = transpose(
        key_layer, shape=shape[:-1] + [num_attention_heads, size_per_head])

    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

    if use_relative_position:
        # `relation_keys` = [F|T, F|T, H]
        relations_keys = model_utils._generate_relative_positions_embeddings(
            shape[1], size_per_head, 64,
            cache=False)
        relations_keys = tf.saturate_cast(relations_keys, tf.float32)
        # query_layer_t is [F, B, N, H]
        query_layer_t = tf.transpose(query_layer, [2, 0, 1, 3])
        # query_layer_r is [F, B * N, H]
        query_layer_r = tf.reshape(query_layer_t, [shape[1], shape[0] * num_attention_heads, size_per_head])
        # key_position_scores is [F, B * N, F|T]
        key_position_scores = tf.matmul(query_layer_r, relations_keys, transpose_b=True)
        # key_position_scores_r is [F, B , N, F|T]
        key_position_scores_r = tf.reshape(key_position_scores,
                                           [shape[1], shape[0], num_attention_heads, shape[1]])
        # key_position_scores_r_t is [B, N, F, F|T]
        key_position_scores_r_t = tf.transpose(key_position_scores_r, [1, 2, 0, 3])
        attention_scores = attention_scores + key_position_scores_r_t

    # 对scores进行缩放，系数为单头size的平方
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))
    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        # 为了维度对齐，先进行维度扩充
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # 保留mask部分的score
        attention_scores += adder

    attention_probs = tf.nn.softmax(attention_scores)

    attention_probs = model_utils.dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    # [bz * seq_len, num_heads * one_head_size] --> [bz, seq_len, num_heads, one_head_size]
    value_layer = transpose(
        value_layer,
        shape=shape[:-1] + [num_attention_heads, size_per_head])

    # `context_layer` = [B, N, F, H]
    # q 和 k 点积得到的score与v相乘，得到context
    # [B, N, F, T] * [B, N, T, H] = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    if use_relative_position:
        # `relation_values` = [F|T, F|T, H]
        relations_values = model_utils._generate_relative_positions_embeddings(
            shape[1], size_per_head, 64, cache=False)
        relations_values = tf.saturate_cast(relations_values, tf.float32)
        # attention_probs_t is [F, B, N, T]
        attention_probs_t = tf.transpose(attention_probs, [2, 0, 1, 3])
        # attention_probs_r is [F, B * N, T]
        attention_probs_r = tf.reshape(attention_probs_t,
                                       [shape[1], shape[0] * num_attention_heads, shape[1]])
        # key_position_scores is [F, B * N, H]
        value_position_scores = tf.matmul(attention_probs_r, relations_values, transpose_b=False)
        # value_position_scores_r is [F, B , N, H]
        value_position_scores_r = tf.reshape(value_position_scores,
                                             [shape[1], shape[0], num_attention_heads, size_per_head])
        # value_position_scores_r_t is [B, N, F, H]
        value_position_scores_r_t = tf.transpose(value_position_scores_r, [1, 2, 0, 3])
        # attention_scores = attention_scores + value_position_scores_r_t
        context_layer = context_layer + value_position_scores_r_t

    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    context_layer = tf.reshape(
        context_layer,
        shape=shape[:2] + [num_attention_heads * size_per_head])

    if do_return_attentions_probs:
        outputs = (context_layer, attention_probs)
    else:
        outputs = (context_layer,)
    return outputs


def attention_output_layer(
        input_tensor,
        hidden_size,
        initializer_range=0.02,
        hidden_dropout_prob=0.1
):
    '''
    输出层, 用于attention输出和transformer输出
    包括一层dense，hidden dropout
    :param input_tensor:
    :param hidden_size:
    :param initializer_range:
    :param hidden_dropout_prob:
    :return:
    '''
    layer_output = tf.layers.dense(
        input_tensor,
        hidden_size,
        kernel_initializer=model_utils.create_initializer(initializer_range))
    layer_output = model_utils.dropout(layer_output, hidden_dropout_prob)
    return layer_output


def intermediate_layer(
        attention_output,
        intermediate_size,
        intermediate_act_fn,
        initializer_range=0.02):
    '''
    transformer中间层，也就是将transformer的输出进行一个全连接操作
    :param attention_output: transformer输出
    :param intermediate_size: dense输出维度
    :param intermediate_act_fn: 激活函数
    :param initializer_range:
    :return:
    '''
    intermediate_output = tf.layers.dense(
        attention_output,
        intermediate_size,
        activation=intermediate_act_fn,
        kernel_initializer=model_utils.create_initializer(initializer_range))
    return intermediate_output


def pooler_layer(
        sequence_output,
        hidden_size,
        initializer_range=0.02):
    """
    bert pooler层，取cls token向量非线性映射
    :param sequence_output:
    :param hidden_size:
    :param initializer_range:
    :return:
    """
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    pooled_output = tf.layers.dense(
        first_token_tensor,
        hidden_size,
        activation=tf.tanh,
        kernel_initializer=model_utils.create_initializer(initializer_range))
    return pooled_output


def crf_layer(input_tensor,
              num_labels,
              labels=None,
              lengths=None,
              scope_name='CRF'):
    """
    crf 层 ， labels为None就返回解码id和trans矩阵，不为None就在前面加上损失
    :param input_tensor:
    :param num_labels:
    :param labels:
    :param lengths:
    :param scope_name:
    :return:
    """
    with tf.variable_scope(scope_name):
        trans = tf.get_variable(
            "transitions",
            shape=[num_labels, num_labels],
            initializer=model_utils.create_initializer(0.02))

        pred_ids, _ = crf.crf_decode(
            potentials=input_tensor,
            transition_params=trans,
            sequence_length=lengths)

        outputs = (pred_ids, trans)
        if labels is not None:
            log_likelihood, trans = crf.crf_log_likelihood(
                inputs=input_tensor,
                tag_indices=labels,
                transition_params=trans,
                sequence_lengths=lengths)

            loss = tf.reduce_mean(-log_likelihood)
            outputs = (loss,) + outputs
    return outputs


def conv2d_layer(
        input_tensor, filter_shape: List[int],
        strides=None, padding="VALID", act='relu',
        initializer_range=0.1):
    """
    二维卷积，一般用于text cnn
    :param input_tensor:
    :param filter_shape:
    :param strides:
    :param padding:
    :param act:
    :param initializer_range:
    :return:
    """
    if strides is None:
        strides = [1, 1, 1, 1]
    W = tf.get_variable(
        name='kernel', shape=filter_shape,
        initializer=model_utils.create_initializer(initializer_range))
    b = tf.get_variable(
        name='bias', shape=[filter_shape[-1]],
        initializer=model_utils.create_initializer(initializer_range))
    output = tf.nn.conv2d(input_tensor, W, strides=strides, padding=padding)
    output = tf.nn.bias_add(output, b)
    act_fn = activations.get_activation(act)
    if act_fn is not None:
        output = act_fn(output)
    return output


def max_pooling_layer(
        input_tensor, ksize,
        strides=None, padding="VALID",
        name='max_pool'):
    """
    最大池化
    :param input_tensor:
    :param ksize:
    :param strides:
    :param padding:
    :param name:
    :return:
    """
    if strides is None:
        strides = [1, 1, 1, 1]

    # 支持动态大小的池化
    output = gen_nn_ops.max_pool_v2(
        input_tensor,
        ksize=ksize,
        strides=strides,
        padding=padding,
        name=name
    )
    # output = tf.nn.max_pool(
    #     input_tensor,
    #     ksize=ksize,
    #     strides=strides,
    #     padding=padding,
    #     name=name
    # )
    return output


def bi_rnn_layer(
        input_tensor, rnn_type, hidden_units,
        hidden_dropout, initializer_range=0.02):
    """
    双向rnn
    :param input_tensor:
    :param rnn_type:
    :param hidden_units:
    :param hidden_dropout:
    :param initializer_range:
    :return:
    """
    cell_map = {
        'lstm': tf.nn.rnn_cell.LSTMCell, 'gru': tf.nn.rnn_cell.GRUCell
    }

    fw_cell = cell_map[rnn_type](
        hidden_units, initializer=model_utils.create_initializer(initializer_range),
        name=rnn_type + '_fw_cell')
    fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=(1 - hidden_dropout))
    bw_cell = cell_map[rnn_type](
        hidden_units, initializer=model_utils.create_initializer(initializer_range),
        name=rnn_type + '_bw_cell')
    bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=(1 - hidden_dropout))

    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
    # output_fw, output_bw，两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
    # 二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
    (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        inputs=input_tensor,
        dtype=tf.float32)
    return output_fw, output_bw, state_fw, state_bw


def attention_pooling_layer(
        input_tensor, initializer_range=0.1):
    """
    加权池化
    :param input_tensor:
    :param initializer_range:
    :return:
    """
    shape = model_utils.get_shape_list(input_tensor, expected_rank=3)
    seq_len, hidden_size = shape[1], shape[2]

    # 初始化一个权重向量，是可训练的参数
    W = tf.get_variable(shape=[hidden_size],
                        initializer=model_utils.create_initializer(initializer_range))

    # 对输入tensor用激活函数做非线性转换
    M = tf.tanh(input_tensor)

    # 对W和M做矩阵运算，W=[batch_size, seq_len, hidden_size]，
    # 计算前做维度转换成[batch_size * seq_len, hidden_size]
    # newM = [batch_size, seq_len, 1]，每一个时间步的输出由向量转换成一个数字
    newM = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

    # 对newM做维度转换成[batch_size, seq_len]
    restoreM = tf.reshape(newM, [-1, seq_len])

    # 用softmax做归一化处理[batch_size, seq_len]
    attention_score = tf.nn.softmax(restoreM)

    # 利用求得的attention_score的值对H进行加权求和，用矩阵运算直接操作
    r = tf.matmul(tf.transpose(input_tensor, [0, 2, 1]),
                  tf.reshape(attention_score, [-1, seq_len, 1]))

    # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
    sequeezeR = tf.reshape(r, [-1, hidden_size])
    output = tf.tanh(sequeezeR)
    return output, attention_score


def mlm_layer(config, sequence_output, embedding_table, mlm_positions, scope='cls/predictions'):
    if mlm_positions is not None:
        sequence_output = model_utils.gather_indexes(sequence_output, mlm_positions)
    with tf.variable_scope(scope):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                sequence_output,
                units=config.embedding_size,
                activation=activations.get_activation(config.hidden_act),
                kernel_initializer=model_utils.create_initializer(
                    config.initializer_range))
            input_tensor = model_utils.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
    return logits


def seq_rel_weight(config, pooled_output, scope='cls/seq_relationship', rel_num=2):
    with tf.variable_scope(scope):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[rel_num, config.hidden_size],
            initializer=model_utils.create_initializer(
                config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[rel_num], initializer=tf.zeros_initializer())

        logits = tf.matmul(pooled_output, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
    return logits
