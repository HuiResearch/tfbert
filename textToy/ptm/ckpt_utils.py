# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: ckpt_utils.py
@date: 2020/09/09
"""
import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import os
import collections
import re
import six
from typing import List


def get_bert_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def get_albert_assignment_map_from_checkpoint(tvars, init_checkpoint, num_of_group=0):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    init_vars = tf.train.list_variables(init_checkpoint)
    init_vars_name = [name for (name, _) in init_vars]

    if num_of_group > 0:
        assignment_map = []
        for gid in range(num_of_group):
            assignment_map.append(collections.OrderedDict())
    else:
        assignment_map = collections.OrderedDict()

    for name in name_to_variable:
        if name in init_vars_name:
            tvar_name = name
        elif (re.sub(r"/group_\d+/", "/group_0/",
                     six.ensure_str(name)) in init_vars_name and
              num_of_group > 1):
            tvar_name = re.sub(r"/group_\d+/", "/group_0/", six.ensure_str(name))
        elif (re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
              in init_vars_name and num_of_group > 1):
            tvar_name = re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
        elif (re.sub(r"/attention_\d+/", "/attention_1/", six.ensure_str(name))
              in init_vars_name and num_of_group > 1):
            tvar_name = re.sub(r"/attention_\d+/", "/attention_1/",
                               six.ensure_str(name))
        else:
            continue
        if num_of_group > 0:
            group_matched = False
            for gid in range(1, num_of_group):
                if (("/group_" + str(gid) + "/" in name) or
                        ("/ffn_" + str(gid) + "/" in name) or
                        ("/attention_" + str(gid) + "/" in name)):
                    group_matched = True
                    tf.logging.info("%s belongs to %dth", name, gid)
                    assignment_map[gid][tvar_name] = name
            if not group_matched:
                assignment_map[0][tvar_name] = name
        else:
            assignment_map[tvar_name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[six.ensure_str(name) + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def search_layer(layer_name):
    '''
    根据 名称查找 tensor
    :param layer_name:
    :return:
    '''
    tvars = tf.trainable_variables()
    for var in tvars:
        if layer_name in var.name:
            return var


def init_checkpoints(init_checkpoint, model_type, print_vars=True):
    model_type = model_type.lower()

    fct_map = {
        'bert': get_bert_assignment_map_from_checkpoint,
        'albert': get_albert_assignment_map_from_checkpoint,
        'nezha': get_bert_assignment_map_from_checkpoint,
        'electra': get_bert_assignment_map_from_checkpoint,
        'wobert': get_bert_assignment_map_from_checkpoint
    }

    if model_type not in fct_map:
        raise ValueError("Unsupported {}, you can choose one of {}"
                         "".format(model_type, '、'.join(list(fct_map.keys()))))

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = fct_map[model_type](tvars,
                                                                           init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if print_vars:
        tf.logging.info("  **** Trainable Variables ****")
        for var in tvars:
            if var.name not in initialized_variable_names:
                init_string = ", *NOT INIT FROM CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)


def init_global_variables(session, init_checkpoint, model_type, print_vars=True):
    '''
    初始化模型参数
    :param session: 当前的会话
    :param init_checkpoint: 模型地址
    :param model_type: 模型类型：bert、nezha、albert等
    :param print_vars: 是否打印日志
    :return:
    '''
    init_checkpoints(init_checkpoint, print_vars=print_vars, model_type=model_type)
    session.run(tf.global_variables_initializer())


def get_save_vars():
    """
    剔除模型中优化这些无用变量
    :return:
    """
    t_vars = tf.trainable_variables()
    var_list = []
    waste_name = ['global_step', 'adam',  # for bert
                  'lamb', 'bad_steps', 'good_steps', 'loss_scale',  # for nezha
                  ]
    for var in t_vars:
        if not any(n in var.name for n in waste_name):
            var_list.append(var)
    return var_list


def clean_bert_model(model_file, save_file, remove_ori_model=False, waste_name_: List[str] = None):
    '''
    将已保存的bert系列模型的优化器参数去掉
    :param model_file:  原始ckpt文件
    :param save_file: 处理后模型保存文件
    :param remove_ori_model: 是否删除原来的模型文件
    :param waste_name_: 自定义去除参数名
    :return:
    '''
    tf.reset_default_graph()
    var_list = tf.train.list_variables(model_file)
    var_values, var_dtypes = {}, {}

    waste_name = ['global_step', 'adam', 'Adam',  # for bert
                  'lamb', 'bad_steps', 'good_steps', 'loss_scale',  # for nezha
                  ]
    if isinstance(waste_name, list):
        waste_name.extend(waste_name_)

    for (name, shape) in var_list:
        if not any(n in name for n in waste_name):
            var_values[name] = None

    reader = contrib.framework.load_checkpoint(model_file)
    for name in var_values:
        tensor = reader.get_tensor(name)
        var_dtypes[name] = tensor.dtype
        var_values[name] = tensor

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        tf_vars = [
            tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
            for v in var_values
        ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]

    saver = tf.train.Saver(tf.all_variables())

    # 去除原本的模型文件
    if remove_ori_model:
        dir, filename = os.path.split(model_file)
        for file in os.listdir(dir):
            file = os.path.join(dir, file)
            if model_file in file:
                os.remove(file)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                               six.iteritems(var_values)):
            sess.run(assign_op, {p: value})

        # Use the built saver to save the averaged checkpoint.
        saver.save(sess, save_file)
