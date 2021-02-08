# -*- coding:utf-8 -*-
# @FileName  :schedule.py
# @Time      :2021/1/31 19:56
# @Author    :huanghui
import tensorflow.compat.v1 as tf


def lr_schedule(init_lr,
                num_train_steps,
                num_warmup_steps,
                decay_method='poly',
                optimizer_type='adamw'):
    '''
    线性学习率， 在warmup steps之前：lr = global_step/num_warmup_steps * init_lr
    :param init_lr:
    :param num_train_steps:
    :param num_warmup_steps:
    :param decay_method:  学习率衰减方式，可选择 poly、cos
    :param optimizer_type:
    :return:
    '''

    global_step = tf.train.get_or_create_global_step()

    # avoid step change in learning rate at end of warmup phase
    if optimizer_type == "adamw":
        power = 1.0
        decayed_learning_rate_at_crossover_point = init_lr * (
                (1.0 - float(num_warmup_steps) / float(num_train_steps)) ** power)
    else:
        power = 0.5
        decayed_learning_rate_at_crossover_point = init_lr

    adjusted_init_lr = init_lr * (init_lr / decayed_learning_rate_at_crossover_point)
    init_lr = tf.constant(value=adjusted_init_lr, shape=[], dtype=tf.float32)

    # increase the learning rate linearly
    if num_warmup_steps > 0:
        warmup_lr = (tf.cast(global_step, tf.float32)
                     / tf.cast(num_warmup_steps, tf.float32)
                     * init_lr)  # 线性增长
    else:
        warmup_lr = 0.0

    # decay the learning rate
    if decay_method == "poly":
        decay_lr = tf.train.polynomial_decay(
            init_lr,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=power,
            cycle=False)
    elif decay_method == "cos":
        decay_lr = tf.train.cosine_decay(
            init_lr,
            global_step,
            num_train_steps,
            alpha=0.0)
    else:
        raise ValueError(decay_method)

    learning_rate = tf.where(global_step < num_warmup_steps,
                             warmup_lr, decay_lr)

    return learning_rate
