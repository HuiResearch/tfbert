# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: transformer.py
@date: 2020/09/08
"""

import tensorflow.compat.v1 as tf
import math
from .utils import (get_shape_list,
                    reshape_to_matrix,
                    create_initializer,
                    dropout,
                    layer_norm,
                    _generate_relative_positions_embeddings
                    )
from .utils import gelu


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    use_relative_position=False,  # nezha使用的相对位置
                    compute_type=tf.float32,  # 使用16还是32位float
                    do_return_attentions_probs=False):

    def transpose_for_scores(input_tensor,
                             batch_size,
                             num_attention_heads,
                             seq_length,
                             width):
        # 将输入tensor进行转置
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        # 将 tensor 转为 bz, num_heads, seq_len, width
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    # 保证 shape 一致
    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # 将 tensor 转为二维张量，bz * seq_len，width
    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    # [bz* seq_len, width] --> [bz * seq_len, num_heads * one_head_size]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    # [bz* seq_len, width] --> [bz * seq_len, num_heads * one_head_size]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    # [bz* seq_len, width] --> [bz * seq_len, num_heads * one_head_size]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    # [bz * seq_len, num_heads * one_head_size] --> [bz, num_heads, seq_len, one_head_size]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    # [bz * seq_len, num_heads * one_head_size] --> [bz, num_heads, seq_len, one_head_size]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    # [B, N, F, H] * [B, N, T, H] = [B, N, F, T]
    # 得到self attention score
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

    if use_relative_position:
        assert from_seq_length == to_seq_length
        max_relative_position = 64
        # `relation_keys` = [F|T, F|T, H]
        relations_keys = _generate_relative_positions_embeddings(
            to_seq_length, size_per_head, max_relative_position,
            cache=False)
        relations_keys = tf.saturate_cast(relations_keys, compute_type)
        # query_layer_t is [F, B, N, H]
        query_layer_t = tf.transpose(query_layer, [2, 0, 1, 3])
        # query_layer_r is [F, B * N, H]
        query_layer_r = tf.reshape(query_layer_t, [from_seq_length, batch_size * num_attention_heads, size_per_head])
        # key_position_scores is [F, B * N, F|T]
        key_position_scores = tf.matmul(query_layer_r, relations_keys, transpose_b=True)
        # key_position_scores_r is [F, B , N, F|T]
        key_position_scores_r = tf.reshape(key_position_scores,
                                           [from_seq_length, batch_size, num_attention_heads, from_seq_length])
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

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    # 归一化后，原始mask为0部分就会趋于0
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    # [bz * seq_len, num_heads * one_head_size] --> [bz, seq_len, num_heads, one_head_size]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    # [bz, seq_len, num_heads, one_head_size] --> [bz, num_heads, seq_len, one_head_size]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    # q 和 k 点积得到的score与v相乘，得到context
    # [B, N, F, T] * [B, N, T, H] = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    if use_relative_position:
        # `relation_values` = [F|T, F|T, H]
        relations_values = _generate_relative_positions_embeddings(
            to_seq_length, size_per_head, max_relative_position,
            cache=False)
        relations_values = tf.saturate_cast(relations_values, compute_type)
        # attention_probs_t is [F, B, N, T]
        attention_probs_t = tf.transpose(attention_probs, [2, 0, 1, 3])
        # attention_probs_r is [F, B * N, T]
        attention_probs_r = tf.reshape(attention_probs_t,
                                       [from_seq_length, batch_size * num_attention_heads, to_seq_length])
        # key_position_scores is [F, B * N, H]
        value_position_scores = tf.matmul(attention_probs_r, relations_values, transpose_b=False)
        # value_position_scores_r is [F, B , N, H]
        value_position_scores_r = tf.reshape(value_position_scores,
                                             [from_seq_length, batch_size, num_attention_heads, size_per_head])
        # value_position_scores_r_t is [B, N, F, H]
        value_position_scores_r_t = tf.transpose(value_position_scores_r, [1, 2, 0, 3])
        # attention_scores = attention_scores + value_position_scores_r_t
        context_layer = context_layer + value_position_scores_r_t

    # `context_layer` = [B, F, N, H]
    # 将得到的context转为 bz，seq_len, num_heads, one_head_size
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

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
        kernel_initializer=create_initializer(initializer_range))
    layer_output = dropout(layer_output, hidden_dropout_prob)
    return layer_output


def intermediate_layer(attention_output,
                       intermediate_size,
                       intermediate_act_fn,
                       initializer_range=0.02
                       ):
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
        kernel_initializer=create_initializer(initializer_range))
    return intermediate_output


def bert_transformer_layer(
        layer_input,
        batch_size=None,
        seq_length=None,
        attention_mask=None,
        hidden_size=768,  # 隐藏层大小
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn=gelu,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        use_relative_position=False,
        compute_type=tf.float32,
        do_return_attentions_probs=False
):
    """
    bert系列的transformer 层结构，因为命名空间和albert这些不一样，所以单独区分出来
    包括多头自注意力层，注意力输出层，中间层，transformer输出层
    每个输出层都使用残差连接
    :param layer_input:
    :param batch_size:
    :param seq_length:
    :param attention_mask:
    :param hidden_size:
    :param num_attention_heads:
    :param intermediate_size:
    :param intermediate_act_fn:
    :param hidden_dropout_prob:
    :param attention_probs_dropout_prob:
    :param initializer_range:
    :param use_relative_position: 是否使用相对位置，针对nezha模型设定
    :param compute_type:  float位数
    :param do_return_attentions_probs: 是否返回attention scores
    :return:
    """
    attention_head_size = int(hidden_size / num_attention_heads)
    with tf.variable_scope("attention"):
        attention_heads = []
        # 多头自注意力层
        with tf.variable_scope("self"):
            attention_layer_outputs = attention_layer(
                from_tensor=layer_input,
                to_tensor=layer_input,
                attention_mask=attention_mask,
                num_attention_heads=num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                do_return_2d_tensor=True,
                batch_size=batch_size,
                from_seq_length=seq_length,
                to_seq_length=seq_length,
                use_relative_position=use_relative_position,
                compute_type=compute_type,
                do_return_attentions_probs=do_return_attentions_probs
            )
            attention_head = attention_layer_outputs[0]
            attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
            attention_output = attention_heads[0]
        else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        # attention输出层
        with tf.variable_scope("output"):
            attention_output = attention_output_layer(
                input_tensor=attention_output,
                hidden_size=hidden_size,
                initializer_range=initializer_range,
                hidden_dropout_prob=hidden_dropout_prob
            )
            attention_output = layer_norm(attention_output + layer_input)

    # transformer中间层
    # The activation is only applied to the "intermediate" hidden layer.
    with tf.variable_scope("intermediate"):
        intermediate_output = intermediate_layer(
            attention_output,
            intermediate_size,
            intermediate_act_fn,
            initializer_range
        )

    # Down-project back to `hidden_size` then add the residual.
    # transformer 输出层
    with tf.variable_scope("output"):
        layer_output = attention_output_layer(
            input_tensor=intermediate_output,
            hidden_size=hidden_size,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob
        )
        layer_output = layer_norm(layer_output + attention_output)
    if do_return_attentions_probs:
        outputs = (layer_output, attention_layer_outputs[1])
    else:
        outputs = (layer_output,)
    return outputs


def albert_transformer_layer(
        layer_input,
        batch_size=None,
        seq_length=None,
        attention_mask=None,
        hidden_size=768,  # 隐藏层大小
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn=gelu,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        use_relative_position=False,
        compute_type=tf.float32,
        do_return_attentions_probs=False
):
    """
    albert的transformer layer，主要是命名空间不一致，所以单独区分，本来可以通过map进行变量映射
    但是为了兼容原始albert模型结构，还是重复写一个
    :param layer_input:
    :param batch_size:
    :param seq_length:
    :param attention_mask:
    :param hidden_size:
    :param num_attention_heads:
    :param intermediate_size:
    :param intermediate_act_fn:
    :param hidden_dropout_prob:
    :param attention_probs_dropout_prob:
    :param initializer_range:
    :param use_relative_position:
    :param compute_type:
    :param do_return_attentions_probs:
    :return:
    """
    attention_head_size = int(hidden_size / num_attention_heads)
    with tf.variable_scope("attention_1"):
        attention_heads = []
        with tf.variable_scope("self"):
            attention_layer_outputs = attention_layer(
                from_tensor=layer_input,
                to_tensor=layer_input,
                attention_mask=attention_mask,
                num_attention_heads=num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                do_return_2d_tensor=True,
                batch_size=batch_size,
                from_seq_length=seq_length,
                to_seq_length=seq_length,
                use_relative_position=use_relative_position,
                compute_type=compute_type,
                do_return_attentions_probs=do_return_attentions_probs
            )
            attention_head = attention_layer_outputs[0]
            attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
            attention_output = attention_heads[0]
        else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
            attention_output = attention_output_layer(
                input_tensor=attention_output,
                hidden_size=hidden_size,
                initializer_range=initializer_range,
                hidden_dropout_prob=hidden_dropout_prob
            )

    # albert 的 layer norm 所在命名空间 和 bert 不一致
    attention_output = layer_norm(attention_output + layer_input)
    with tf.variable_scope("ffn_1"):
        with tf.variable_scope("intermediate"):
            intermediate_output = intermediate_layer(
                attention_output,
                intermediate_size,
                intermediate_act_fn,
                initializer_range
            )

            with tf.variable_scope("output"):
                layer_output = attention_output_layer(
                    input_tensor=intermediate_output,
                    hidden_size=hidden_size,
                    initializer_range=initializer_range,
                    hidden_dropout_prob=hidden_dropout_prob
                )

    # albert 的 layer norm 所在命名空间 和 bert 不一致
    layer_output = layer_norm(layer_output + attention_output)
    if do_return_attentions_probs:
        outputs = (layer_output, attention_layer_outputs[1])
    else:
        outputs = (layer_output,)
    return outputs


def pooler_layer(sequence_output,
                 hidden_size,
                 initializer_range=0.02
                 ):
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    pooled_output = tf.layers.dense(
        first_token_tensor,
        hidden_size,
        activation=tf.tanh,
        kernel_initializer=create_initializer(initializer_range))
    return pooled_output


def dense_layer_2d(input_tensor,
                   output_size,
                   initializer_range,
                   activation,
                   use_einsum,
                   name=None):
    """A dense layer with 2D kernel.
    Args:
      input_tensor: Float tensor with rank 3.
      output_size: The size of output dimension.
      initializer_range: Kernel initializer range.
      activation: Activation function.
      use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers.
      name: The name scope of this layer.
    Returns:
      float logits Tensor.
    """

    input_shape = get_shape_list(input_tensor)
    hidden_size = input_shape[2]
    with tf.variable_scope(name):
        w = tf.get_variable(
            name="kernel",
            shape=[hidden_size, output_size],
            initializer=create_initializer(initializer_range))
        b = tf.get_variable(
            name="bias", shape=[output_size], initializer=tf.zeros_initializer)
        if use_einsum:
            ret = tf.einsum("BFH,HO->BFO", input_tensor, w)
        else:
            ret = tf.matmul(input_tensor, w)
        ret += b
    if activation is not None:
        return activation(ret)
    else:
        return ret
