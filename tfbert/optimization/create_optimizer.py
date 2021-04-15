# -*- coding:utf-8 -*-
# @FileName  :create_optimizer.py
# @Time      :2021/1/31 19:58
# @Author    :huanghui
import tensorflow.compat.v1 as tf
from .adamw import AdamWeightDecayOptimizer
from .lamb import LAMBOptimizer
from .schedule import lr_schedule


def create_optimizer(
        learning_rate,
        num_train_steps=None,
        num_warmup_steps=None,
        optimizer_type='adamw',
        epsilon=1e-6,
        momentum=0.,
        weight_decay=0.01,
        decay_method='poly',
        mixed_precision=False,
        init_loss_scale=2 ** 32
):
    if decay_method is not None and num_train_steps is not None and num_warmup_steps is not None:
        num_train_steps = int(num_train_steps)
        num_warmup_steps = int(num_warmup_steps)
        learning_rate = lr_schedule(
            learning_rate, num_train_steps, num_warmup_steps,
            decay_method=decay_method, optimizer_type=optimizer_type
        )

    if optimizer_type == 'adamw':
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=epsilon,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
        )
    elif optimizer_type == 'lamb':
        optimizer = LAMBOptimizer(
            learning_rate,
            weight_decay_rate=weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=epsilon,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
        )
    elif optimizer_type == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=epsilon)
    elif optimizer_type == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate
        )
    elif optimizer_type == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=learning_rate,
            rho=0.95,
            epsilon=epsilon,
        )
    elif optimizer_type == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=learning_rate,
            initial_accumulator_value=0.1
        )
    elif optimizer_type == 'rmsp':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            decay=0.9,
            momentum=momentum,
            epsilon=epsilon,
        )
    else:
        raise ValueError('Unsupported optimizer option: %s' % optimizer_type)

    if mixed_precision:
        loss_scaler = tf.train.experimental.DynamicLossScale(
            initial_loss_scale=init_loss_scale, increment_period=1000,
            multiplier=2.0)
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scaler)
        loss_scale_value = tf.identity(loss_scaler(), name="loss_scale")
    return optimizer


def create_train_op(
        optimizer,
        grads_and_vars,
        max_grad=1.0,
        mixed_precision=False,
        gradient_accumulation_steps=1):
    global_step = tf.train.get_or_create_global_step()

    if gradient_accumulation_steps > 1:
        local_step = tf.get_variable(name="local_step", shape=[], dtype=tf.int32, trainable=False,
                                     initializer=tf.zeros_initializer)
        batch_finite = tf.get_variable(name="batch_finite", shape=[], dtype=tf.bool, trainable=False,
                                       initializer=tf.ones_initializer)
        accum_vars = [tf.get_variable(
            name=tvar.name.split(":")[0] + "/accum",
            shape=tvar.shape.as_list(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer()) for tvar in tf.trainable_variables()]

        reset_step = tf.cast(tf.math.equal(local_step % gradient_accumulation_steps, 0), dtype=tf.bool)
        local_step = tf.cond(reset_step, lambda: local_step.assign(tf.ones_like(local_step)),
                             lambda: local_step.assign_add(1))

        grads_and_vars_and_accums = [(gv[0], gv[1], accum_vars[i]) for i, gv in enumerate(grads_and_vars) if
                                     gv[0] is not None]
        grads, tvars, accum_vars = list(zip(*grads_and_vars_and_accums))

        all_are_finite = tf.reduce_all(
            [tf.reduce_all(tf.is_finite(g)) for g in grads]) if mixed_precision else tf.constant(
            True,
            dtype=tf.bool)
        batch_finite = tf.cond(reset_step,
                               lambda: batch_finite.assign(
                                   tf.math.logical_and(tf.constant(True, dtype=tf.bool), all_are_finite)),
                               lambda: batch_finite.assign(tf.math.logical_and(batch_finite, all_are_finite)))

        # This is how the model was pre-trained.
        # ensure global norm is a finite number
        # to prevent clip_by_global_norm from having a hizzy fit.
        (clipped_grads, _) = tf.clip_by_global_norm(
            grads, clip_norm=max_grad)

        accum_vars = tf.cond(reset_step,
                             lambda: [accum_vars[i].assign(grad) for i, grad in enumerate(clipped_grads)],
                             lambda: [accum_vars[i].assign_add(grad) for i, grad in enumerate(clipped_grads)])

        def update(accum_vars):
            return optimizer.apply_gradients(list(zip(accum_vars, tvars)))

        update_step = tf.identity(
            tf.cast(tf.math.equal(local_step % gradient_accumulation_steps, 0), dtype=tf.bool),
            name="update_step")
        update_op = tf.cond(update_step,
                            lambda: update(accum_vars), lambda: tf.no_op())

        new_global_step = tf.cond(tf.math.logical_and(update_step, batch_finite),
                                  lambda: global_step + 1,
                                  lambda: global_step)
        new_global_step = tf.identity(new_global_step, name='step_update')
        train_op = tf.group(update_op, [global_step.assign(new_global_step)])
    else:
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        grads, tvars = list(zip(*grads_and_vars))
        all_are_finite = tf.reduce_all(
            [tf.reduce_all(tf.is_finite(g)) for g in grads]) if mixed_precision else tf.constant(True,
                                                                                                 dtype=tf.bool)

        # This is how the model was pre-trained.
        # ensure global norm is a finite number
        # to prevent clip_by_global_norm from having a hizzy fit.
        (clipped_grads, _) = tf.clip_by_global_norm(
            grads, clip_norm=max_grad)

        # 这里不要传入global step，adam内部没有对global step累加
        # 而原本adam等tf内置优化器会累加，这样就会造成global step重复增加
        train_op = optimizer.apply_gradients(
            list(zip(clipped_grads, tvars)))

        new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
        new_global_step = tf.identity(new_global_step, name='step_update')

        train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op
