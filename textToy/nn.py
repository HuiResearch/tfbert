# -*- coding: UTF-8 -*-
"""
传统神经网络层
@author: huanghui
@file: nn.py
@date: 2020/09/11
"""
import tensorflow.compat.v1 as tf
from tensorflow.contrib import crf
from .ptm.utils import create_initializer, get_shape_list
from typing import List


class CRF:
    def __init__(self,
                 input_tensor,
                 num_labels,
                 labels=None,
                 lengths=None,
                 scope_name='CRF'):
        with tf.variable_scope(scope_name):
            trans = tf.get_variable(
                "transitions",
                shape=[num_labels, num_labels],
                initializer=create_initializer(0.02))

            if labels is not None:
                log_likelihood, trans = crf.crf_log_likelihood(
                    inputs=input_tensor,
                    tag_indices=labels,
                    transition_params=trans,
                    sequence_lengths=lengths)

                self.loss = tf.reduce_mean(-log_likelihood)

            self.trans = trans

            self.pred_ids, _ = crf.crf_decode(potentials=input_tensor,
                                              transition_params=self.trans,
                                              sequence_length=lengths)


class Conv2D:
    def __init__(self, input_tensor, filter_shape: List[int], strides=None, padding="VALID"):
        if strides is None:
            strides = [1, 1, 1, 1]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="weight")
        b = tf.Variable(tf.constant(0.1, shape=[filter_shape[-1]]), name="bias")
        conv = tf.nn.conv2d(input_tensor, W, strides=strides, padding=padding, name='conv')
        self.output = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")


class MaxPool:
    def __init__(self, input_tensor, ksize: List[int], strides=None, padding="VALID"):
        if strides is None:
            strides = [1, 1, 1, 1]
        self.output = tf.nn.max_pool(
            input_tensor,
            ksize=ksize,
            strides=strides,
            padding=padding,
            name='max_pool'
        )


class BiRNN:
    def __init__(self, rnn_type, input_tensor, hidden_units, dropout):
        cell_map = {
            'lstm': tf.nn.rnn_cell.LSTMCell, 'gru': tf.nn.rnn_cell.GRUCell
        }

        fw_cell = cell_map[rnn_type](hidden_units, initializer=create_initializer(0.02), name=rnn_type + '_fw_cell')
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=(1-dropout))
        bw_cell = cell_map[rnn_type](hidden_units, initializer=create_initializer(0.02), name=rnn_type + '_bw_cell')
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=(1 - dropout))

        # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
        # output_fw, output_bw，两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
        # 二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
        (self.output_fw, self.output_bw), (self.state_fw, self.state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=input_tensor,
            dtype=tf.float32)


# 注意力加权，可替换最大池化这些操作
class Attention:
    def __init__(self, input_tensor):

        shape = get_shape_list(input_tensor, expected_rank=3)
        seq_len, hidden_size = shape[1], shape[2]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.truncated_normal([hidden_size], stddev=0.1))

        # 对输入tensor用激活函数做非线性转换
        M = tf.tanh(input_tensor)

        # 对W和M做矩阵运算，W=[batch_size, seq_len, hidden_size]，
        # 计算前做维度转换成[batch_size * seq_len, hidden_size]
        # newM = [batch_size, seq_len, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, seq_len]
        restoreM = tf.reshape(newM, [-1, seq_len])

        # 用softmax做归一化处理[batch_size, seq_len]
        self.attention_score = tf.nn.softmax(restoreM)

        # 利用求得的attention_score的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(input_tensor, [0, 2, 1]),
                      tf.reshape(self.attention_score, [-1, seq_len, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hidden_size])
        self.output = tf.tanh(sequeezeR)


