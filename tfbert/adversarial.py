# -*- coding: UTF-8 -*-
__author__ = 'huanghui'
__date__ = '2021/5/7 20:31'
__project__ = 'tfbert'

import tensorflow.compat.v1 as tf
from . import utils
import numpy as np


class AdversarialOutput:
    def __init__(self, outputs: dict, grads_and_vars):
        self.outputs = outputs
        self.grads_and_vars = grads_and_vars

    def keys(self):
        return list(self.outputs.keys())

    def __getitem__(self, item):
        return self.outputs[item]


def fgm(model_fn, inputs, optimizer=None, layer_name='word_embeddings', epsilon=0.5):
    """
    FGM对抗训练tensorflow1.x实现
    :param model_fn:
    :param inputs:
    :param optimizer: 优化器
    :param layer_name: 扰动的变量名
    :param epsilon: 扰动参数
    :return:
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        model_outputs = model_fn(inputs, True)
        grads_and_vars = utils.compute_gradients(model_outputs['loss'], optimizer)
    # loss对embedding的梯度
    embedding_gradients, embeddings = utils.find_grad_and_var(grads_and_vars, layer_name)

    r = tf.multiply(epsilon, embedding_gradients / (tf.norm(embedding_gradients) + 1e-9))
    attack_op = embeddings.assign(embeddings + r)
    # restore
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE), tf.control_dependencies([attack_op]):
        adv_outputs = model_fn(inputs, True)
        attack_grad_and_vars = utils.compute_gradients(adv_outputs['loss'], optimizer)
        restore_op = embeddings.assign(embeddings - r)

    # sum up
    with tf.control_dependencies([restore_op]):
        grads_and_vars = utils.average_grads_and_vars([grads_and_vars, attack_grad_and_vars])

    return AdversarialOutput(model_outputs, grads_and_vars)


def pgd(model_fn, inputs, optimizer=None, layer_name='word_embeddings', epsilon=0.05, n_loop=2):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        model_outputs = model_fn(inputs, True)
        grads_and_vars = utils.compute_gradients(model_outputs['loss'], optimizer)
    acc_r = 0.0
    attack_op = tf.no_op()
    for k in range(n_loop):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE), tf.control_dependencies([attack_op]):
            adv_outputs = model_fn(inputs, True)
            attack_grad_and_vars = utils.compute_gradients(adv_outputs['loss'], optimizer)
            embedding_gradients, embeddings = utils.find_grad_and_var(attack_grad_and_vars, layer_name)

            tmp_r = tf.multiply(1 / n_loop, embedding_gradients / (tf.norm(embedding_gradients) + 1e-9))

            norm = tf.norm(acc_r + tmp_r)
            cur_r = tf.cond(norm > epsilon,
                            lambda: (acc_r + tmp_r) * tf.divide(epsilon, norm),
                            lambda: (acc_r + tmp_r))
            r = cur_r - acc_r  # calculate current step
            attack_op = embeddings.assign(embeddings + r)
            acc_r = cur_r

    # restore
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE), tf.control_dependencies([attack_op]):
        attack_outputs = model_fn(inputs, True)
        attack_grad_and_vars = utils.compute_gradients(attack_outputs['loss'], optimizer)
        embedding_gradients, embeddings = utils.find_grad_and_var(attack_grad_and_vars, layer_name)
        restore_op = embeddings.assign(embeddings - acc_r)
    # sum up
    with tf.control_dependencies([restore_op]):
        grads_and_vars = utils.average_grads_and_vars([grads_and_vars, attack_grad_and_vars])
    return AdversarialOutput(model_outputs, grads_and_vars)


def freelb(
        model_fn, inputs, batch_size, max_length,
        optimizer=None, layer_name='word_embeddings',
        epsilon=0.3, n_loop=3):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        model_outputs = model_fn(inputs, True)
        grads_and_vars = utils.compute_gradients(model_outputs['loss'], optimizer)
    # loss对embedding的梯度
    embedding_gradients, embeddings = utils.find_grad_and_var(grads_and_vars, layer_name)
    init_r = tf.get_variable(
        'init_r',
        shape=[batch_size * max_length,
               embeddings.shape.as_list()[-1]],
        initializer=tf.random_uniform_initializer(
            minval=-epsilon, maxval=epsilon),
        trainable=False)
    init_op = tf.variables_initializer([init_r])
    with tf.control_dependencies([init_op]):  # fix perturbation
        # Scale randomly initialized permutation, to make sure norm
        # of `r` is smaller than epsilon.
        r = tf.divide(init_r, tf.norm(init_r, np.inf))
        r = tf.IndexedSlices(values=r,
                             indices=embedding_gradients.indices,
                             dense_shape=embedding_gradients.dense_shape)
        attack_op = embeddings.assign(embeddings + r)
    # attack
    acc_r = r
    all_grads_and_vars = []
    for k in range(n_loop):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE), tf.control_dependencies([attack_op]):
            adv_outputs = model_fn(inputs, True)
            attack_grad_and_vars = utils.compute_gradients(adv_outputs['loss'], optimizer)
            all_grads_and_vars.append(attack_grad_and_vars)
            gradients, _ = utils.find_grad_and_var(attack_grad_and_vars, layer_name)
            tmp_r = tf.multiply(1 / n_loop, gradients / (tf.norm(gradients) + 1e-9))

            # In order not to shuffle the distribution of gradient-
            # induced perturbation, we use norm to scale instead of
            # simply clip the values.
            norm = tf.norm(acc_r + tmp_r)
            cur_r = tf.cond(norm > epsilon,
                            lambda: (acc_r + tmp_r) * tf.divide(epsilon, norm),
                            lambda: (acc_r + tmp_r))
            r = cur_r - acc_r  # calculate current step
            attack_op = embeddings.assign(embeddings + r)
            acc_r = cur_r
    # restore
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE), tf.control_dependencies([attack_op]):
        attack_outputs = model_fn(inputs, True)
        attack_grad_and_vars = utils.compute_gradients(attack_outputs['loss'], optimizer)

        all_grads_and_vars.append(attack_grad_and_vars)
        restore_op = embeddings.assign(embeddings - acc_r)

    # sum up
    with tf.control_dependencies([restore_op]):
        grads_and_vars = utils.average_grads_and_vars(all_grads_and_vars)
    return AdversarialOutput(model_outputs, grads_and_vars)
