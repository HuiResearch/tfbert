# -*- coding:utf-8 -*-
# @FileName  :model_utils.py
# @Time      :2021/1/31 15:35
# @Author    :huanghui
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
import tensorflow.contrib.layers as contrib_layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops
import numpy
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn
import six
import numpy as np


def create_weight(shape, var_name, initializer_range=0.02):
    return tf.get_variable(
        name=var_name,
        shape=shape,
        initializer=create_initializer(initializer_range))


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.layers.dropout(input_tensor, dropout_prob, training=None)
    return output


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def fused_layer_norm(inputs,
                     center=True,
                     scale=True,
                     activation_fn=None,
                     reuse=None,
                     variables_collections=None,
                     outputs_collections=None,
                     trainable=True,
                     begin_norm_axis=1,
                     begin_params_axis=-1,
                     scope=None,
                     use_fused_batch_norm=False):
    with tf.variable_scope(
            scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.shape
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        if begin_norm_axis < 0:
            begin_norm_axis = inputs_rank + begin_norm_axis
        if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
            raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                             'must be < rank(inputs) (%d)' %
                             (begin_params_axis, begin_norm_axis, inputs_rank))
        params_shape = inputs_shape[begin_params_axis:]
        if not params_shape.is_fully_defined():
            raise ValueError(
                'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
                (inputs.name, begin_params_axis, inputs_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                              'beta')
            beta = variables.model_variable(
                'beta',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.zeros_initializer(),
                collections=beta_collections,
                trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(
                variables_collections, 'gamma')
            gamma = variables.model_variable(
                'gamma',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.ones_initializer(),
                collections=gamma_collections,
                trainable=trainable)
        if use_fused_batch_norm:
            # get static TensorShape if fully defined,
            # otherwise retrieve shape tensor
            norm_shape = inputs.shape[begin_norm_axis:]
            if norm_shape.is_fully_defined():
                bn_shape = [1, -1, 1, numpy.prod(norm_shape.as_list())]
            else:
                norm_shape = tf.shape(inputs)[begin_norm_axis:]
                bn_shape = [1, -1, 1, tf.reduce_prod(norm_shape)]
            if inputs.get_shape().is_fully_defined():
                outputs_shape = inputs.get_shape()
            else:
                outputs_shape = tf.shape(inputs)
            inputs = array_ops.reshape(inputs, bn_shape)
            if inputs.get_shape().is_fully_defined():
                # static inputs TensorShape fully defined after reshape.
                ones = array_ops.ones(inputs.get_shape()[1], dtype=dtypes.float32)
                zeros = array_ops.zeros(inputs.get_shape()[1], dtype=dtypes.float32)
            else:
                # static inputs TensorShape NOT fully defined after reshape.
                # must use dynamic shape, which means these input tensors
                # have to be created at runtime, which causes a slowdown.
                scale_shape = tf.shape(inputs)[1]
                ones = array_ops.ones(scale_shape, dtype=dtypes.float32)
                zeros = array_ops.zeros(scale_shape, dtype=dtypes.float32)
            outputs, mean, variance = nn.fused_batch_norm(
                inputs,
                ones, zeros,
                epsilon=1e-4,
                data_format="NCHW")
            outputs = array_ops.reshape(outputs, outputs_shape)
            if center and scale:
                outputs = outputs * gamma + beta
            elif center:
                outputs = outputs + beta
            elif scale:
                outputs = outputs * gamma
        else:
            # Calculate the moments on the last axis (layer activations).
            norm_axes = list(range(begin_norm_axis, inputs_rank))
            mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
            # Compute layer normalization using the batch_normalization function.
            variance_epsilon = 1e-4
            outputs = nn.batch_normalization(
                inputs,
                mean,
                variance,
                offset=beta,
                scale=gamma,
                variance_epsilon=variance_epsilon)
            outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    if input_tensor.dtype == tf.float16:
        return fused_layer_norm(
            inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name,
            use_fused_batch_norm=True)

    else:
        return contrib_layers.layer_norm(
            inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name='LayerNorm'):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name=name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def create_bert_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

        Args:
          from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
          to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
          float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


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


# nezha模型创建相对位置矩阵
def _generate_relative_positions_matrix(length, max_relative_position,
                                        cache=False):
    """Generates matrix of relative positions between inputs."""
    if not cache:
        range_vec = tf.range(length)
        range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
        distance_mat = range_mat - tf.transpose(range_mat)
    else:
        distance_mat = tf.expand_dims(tf.range(-length + 1, 1, 1), 0)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                            max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def _generate_relative_positions_embeddings(length, depth,
                                            max_relative_position,
                                            cache=False):
    """
    Generates tensor of size [1 if cache else length, length, depth].
    example:
        # `relation_keys` = [F|T, F|T, H]
           relations_keys = _generate_relative_positions_embeddings(
        to_seq_length, size_per_head, max_relative_position, "relative_positions_keys",
        cache=False)
      relations_keys = tf.saturate_cast(relations_keys, compute_type)
    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
      length = to_seq_length
      depth = size_per_head
      max_relative_position
      name = "relative_positions_keys"
    """
    relative_positions_matrix = _generate_relative_positions_matrix(
        length, max_relative_position, cache=cache)
    vocab_size = max_relative_position * 2 + 1
    # Generates embedding for each relative position of dimension depth.
    embeddings_table = np.zeros([vocab_size,
                                 depth])

    for pos in range(vocab_size):
        for i in range(depth // 2):
            embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
            embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

    embeddings_table_tensor = tf.convert_to_tensor(embeddings_table, tf.float32)
    flat_relative_positions_matrix = tf.reshape(relative_positions_matrix, [-1])

    one_hot_relative_positions_matrix = tf.one_hot(flat_relative_positions_matrix, depth=vocab_size)

    embeddings = tf.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)

    my_shape = get_shape_list(relative_positions_matrix)
    my_shape.append(depth)

    embeddings = tf.reshape(embeddings, my_shape)
    return embeddings


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def einsum_via_matmul(input_tensor, w, num_inner_dims):
    """Implements einsum via matmul and reshape ops.
    Args:
      input_tensor: float Tensor of shape [<batch_dims>, <inner_dims>].
      w: float Tensor of shape [<inner_dims>, <outer_dims>].
      num_inner_dims: int. number of dimensions to use for inner products.
    Returns:
      float Tensor of shape [<batch_dims>, <outer_dims>].
    """
    input_shape = get_shape_list(input_tensor)
    w_shape = get_shape_list(w)
    batch_dims = input_shape[: -num_inner_dims]
    inner_dims = input_shape[-num_inner_dims:]
    outer_dims = w_shape[num_inner_dims:]
    inner_dim = np.prod(inner_dims)
    outer_dim = np.prod(outer_dims)
    if num_inner_dims > 1:
        input_tensor = tf.reshape(input_tensor, batch_dims + [inner_dim])
    if len(w_shape) > 2:
        w = tf.reshape(w, [inner_dim, outer_dim])
    ret = tf.matmul(input_tensor, w)
    if len(outer_dims) > 1:
        ret = tf.reshape(ret, batch_dims + outer_dims)
    return ret


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
       float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def get_custom_getter(compute_type):
    return float32_variable_storage_getter if compute_type == tf.float16 else None
