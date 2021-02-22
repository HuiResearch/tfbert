# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: loss.py
@date: 2020/09/08
"""
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import array_ops
from ..utils import search_layer


def cross_entropy_loss(logits, targets, depth):
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(targets, depth=depth, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return loss


def loss_with_gradient_penalty(
        loss, epsilon=1,
        layer_name='word_embeddings',
        gradients_fn=None):
    '''
    参考苏神的带梯度惩罚的损失
    :param loss: 原本计算得到的loss
    :param epsilon:
    :param layer_name:
    :param gradients_fn: 梯度计算方法，tf.gradients或者optimizer.compute_gradients
    :return:
    '''
    if gradients_fn is None:
        gradients_fn = tf.gradients
    embeddings = search_layer(layer_name)
    gp = tf.reduce_sum(gradients_fn(loss, [embeddings])[0] ** 2)
    return loss + 0.5 * epsilon * gp


def mlm_loss(logits, targets, depth, label_weights):
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(targets, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=depth, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator
    return loss


def soft_cross_entropy(logits, targets):
    log_probs = tf.nn.log_softmax(logits, dim=-1)
    targets_prob = tf.nn.softmax(targets, dim=-1)
    per_example_loss = -tf.reduce_sum(targets_prob * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return loss


def mse_loss(logits, targets):
    return tf.reduce_mean(tf.square(targets - logits))


def focal_loss(prediction_tensor, target_tensor, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_mean(per_entry_cross_ent)
