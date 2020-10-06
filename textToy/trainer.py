# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: trainer.py
@date: 2020/09/10
"""
import os
import numpy as np
from .gpu_utils import average_grads_and_vars
import tensorflow.compat.v1 as tf
from .ptm.ckpt_utils import init_checkpoints, get_save_vars
from . import MODELS


class Trainer:
    model_name = "model.ckpt"

    def __init__(self,
                 model_type=None,
                 output_types=None,
                 output_shapes=None,
                 device='gpu'):
        """
        tensorflow 模型多卡训练器，
        如需使用多卡，需要设置好环境变量CUDA_VISIBLE_DEVICES=0,1,2,3 (卡由自己定义，这里表示使用0 1 2 3 四块卡)
        用法：
            训练： 1、创建dataset。
                  2、创建一个trainer对象
                  3、初始化一个模型，将trainer的inputs作为模型的输入，is_training也可以导入trainer的
                  4、传入model_fn, trainer.build_model(model_fn)，详情见build_model注释
                  4、配置模型的优化器，得到train_op
                  5、调用trainer.compile()，训练阶段需要将trainer_op传入。
                  6、调用trainer.build_handle()，将dataset分别传入，根据传入的dataset构建相应handle
                  7、如果加载预训练参数，调用trainer.from_pretrained()；如果不加载，调用trainer.init_variables()
                  8、
                     使用trainer.backward()累积梯度，会返回训练loss
                     开始使用trainer.train_step()训练
                     验证调用trainer.eval_step()，会返回compile时的outputs，
                     验证和预测记得先使用trainer.init_iterator()初始化

            预测：1、创建一个trainer对象
                 2、初始化一个模型，将trainer的inputs作为模型的输入，is_training也可以导入trainer的
                 3、传入model_fn, trainer.build_model(model_fn)，详情见build_model注释
                 4、调用trainer.build_handle()，将预测dataset传入
                 5、调用trainer.from_pretrained()加载参数
                 6、开始使用trainer.test_step()预测，返回的是model_fn返回的outputs字段
        :param model_type:
        :param output_types:
        :param output_shapes:
        """
        tf.logging.set_verbosity(tf.logging.INFO)

        # 获取环境变量的devices
        devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if devices is None:
            self.devices = [0]
        else:
            self.devices = list(map(int, devices.split(',')))

        # gpu还是cpu环境
        self.device_type = device

        # 预定义节点
        self.train_op = None
        self.backward_op = None
        self.zero_grad_op = None

        self.loss = None
        self.outputs = []
        self.gradients = None  # loss 梯度，传给优化器使用
        self.variables = None

        # handle 控制接入的是训练、验证或测试dataset
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(self.handle, output_types, output_shapes)
        inputs = iterator.get_next()

        # 分发 dataset
        self.inputs = self.distribute_dataset(inputs)

        # is training 判定 训练预测环境，也就是dropout这些参数
        self.is_training = tf.placeholder(dtype=tf.bool)

        self.model_type = model_type

        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        sess_conf.allow_soft_placement = True
        self.inited = False
        self.session = tf.Session(config=sess_conf)
        self.saver = None
        self.global_step = 0

        self.train_handle = None
        self.dev_handle = None
        self.test_handle = None

        self.train_iterator = None
        self.dev_iterator = None
        self.test_iterator = None

        self.compiled = False

    @classmethod
    def is_gpu_available(cls):
        return tf.test.is_gpu_available()

    @property
    def num_params(self):
        # 参数量，需要配置好图、会话以后用
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        return num_params

    @property
    def num_devices(self):
        # devices 数量
        return len(self.devices)

    def device_index(self, device):
        return self.devices.index(device)

    def distribute_dataset(self, inputs):
        # 切分dataset，为每个device分配数据
        if self.num_devices > 1:
            examples = [{} for _ in range(self.num_devices)]
            for key in inputs.keys():
                vals = tf.split(inputs[key], self.num_devices, 0)
                for device_id in self.devices:
                    device_index = self.device_index(device_id)
                    examples[device_index][key] = vals[device_index]
        else:
            examples = [inputs]
        return examples

    def get_inputs(self, device):
        # 根据device id 找到对应的inputs
        return self.inputs[self.device_index(device)]

    def get_is_training(self):
        return self.is_training

    def check_file(self, filename_or_path):
        if os.path.isdir(filename_or_path):
            ckpt = os.path.join(filename_or_path, self.model_name)
        else:
            ckpt = filename_or_path
        return ckpt

    def from_pretrained(self, model_name_or_path):
        """
        加载参数需要在compile之后
        :param model_name_or_path:
        :return:
        """
        if os.path.isdir(model_name_or_path):
            ckpt = tf.train.latest_checkpoint(model_name_or_path)
            if ckpt is None:
                ckpt = os.path.join(model_name_or_path, self.model_name)
        else:
            ckpt = model_name_or_path
        if self.model_type is not None and self.model_type in MODELS.keys():
            init_checkpoints(ckpt, self.model_type, True)
            self.session.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.session, ckpt)
        self.inited = True
        tf.logging.info("  Load model from {}".format(ckpt))

    def init_variables(self):
        self.session.run(tf.global_variables_initializer())
        self.inited = True
        tf.logging.info("  Inited global variables.")

    def save_pretrained(self, save_path_or_name):
        ckpt = self.check_file(save_path_or_name)
        self.saver.save(self.session, ckpt, global_step=self.global_step)
        tf.logging.info("  Saved model to {}".format(ckpt))

    def build_handle(self,
                     dataset, set_type='train'):
        if set_type == 'train':
            iterator = dataset.make_one_shot_iterator()
        else:
            iterator = dataset.make_initializable_iterator()
        handle = self.session.run(iterator.string_handle())
        if set_type == 'train':
            self.train_handle = handle
            self.train_iterator = iterator
        elif set_type == 'dev':
            self.dev_handle = handle
            self.dev_iterator = iterator
        elif set_type == 'test':
            self.test_handle = handle
            self.test_iterator = iterator
        else:
            raise ValueError("set_type must be train, dev or test")

    def build_model(self, model_fn):
        """
        传入model_fn，也就是 model 构造函数，model_fn只能接收inputs和is_training

        model_fn 返回一个字典结果，训练需要给定loss
        如果需要验证和测试，需要传入outputs，outputs是一个列表
        examples:

        trainer = Trainer(
        model_type, output_types, output_shapes)

        def get_model_fn(model_type, config, num_classes):
            def model_fn(inputs, is_training):   # 接收两个参数，input在一开始的trainer初始化就定义好了
                model = SequenceClassification(
                    model_type=model_type, config=config,
                    num_classes=num_classes, is_training=is_training,
                    **inputs)
                outputs = [model.logits, inputs['label_ids']]
                loss = model.loss
                return {'loss': loss, 'outputs': outputs}

            return model_fn

        trainer.build_model(get_model_fn(model_type, config, num_classes=len(labels)))

        :param model_fn:
        :return:
        """

        tower_losses, tower_grads_and_vars = [], []
        outputs = []

        for i, device in enumerate(self.devices):
            reuse = True if i > 0 else None
            with tf.device('/{}:{}'.format(self.device_type, i)), \
                 tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                output = model_fn(self.get_inputs(device), self.is_training)

                if 'loss' in output:
                    all_vars = tf.trainable_variables()
                    grads = tf.gradients(output['loss'], all_vars)
                    print(self.get_inputs(device))
                    print(grads)
                    grads_and_vars = list(zip(grads, all_vars))

                    tower_losses.append(output['loss'])
                    tower_grads_and_vars.append(grads_and_vars)
                if 'outputs' in output:
                    for i, o in enumerate(output['outputs']):
                        if len(outputs) < i + 1:
                            outputs.append([])
                        outputs[i].append(o)

        loss = None
        grads_and_vars = None
        new_outputs = []
        if self.num_devices > 1:
            if len(tower_losses) > 0:
                loss = tf.add_n(tower_losses) / len(tower_losses)
                grads_and_vars = average_grads_and_vars(tower_grads_and_vars)

            if len(outputs) > 0:
                for out in outputs:
                    new_outputs.append(tf.concat(out, 0))
        else:
            if len(tower_losses) > 0:
                loss = tower_losses[0]
                grads_and_vars = tower_grads_and_vars[0]
            if len(outputs) > 0:
                for out in outputs:
                    new_outputs.append(out[0])
        self.loss = loss
        self.outputs = new_outputs
        if grads_and_vars is not None:
            self.process_grad_and_vars(grads_and_vars)

    def process_grad_and_vars(self, grads_and_vars):
        gradients, variables = zip(*grads_and_vars)
        #
        # self.gradients = [tf.Variable(tf.zeros_like(v), trainable=False) for v in gradients]
        self.gradients = []
        self.variables = []
        for g, v in zip(gradients, variables):
            if g is not None:
                self.gradients.append(tf.Variable(tf.zeros_like(g), trainable=False))
            else:
                self.gradients.append(None)
            self.variables.append(v)

        gradients_accum_ops = []

        for i, grad in enumerate(gradients):
            if grad is not None:
                gradients_accum_ops.append(self.gradients[i].assign_add(grad))

        self.backward_op = tf.group(*gradients_accum_ops, name="backward")  # 梯度累计op

        grads_zero_ops = []
        for gv in self.gradients:
            if gv is not None:
                grads_zero_ops.append(gv.assign(tf.zeros_like(gv)))
        self.zero_grad_op = tf.group(*grads_zero_ops, name='zero_grad')  # 梯度清零 op

    def compile(self, train_op=None, max_checkpoints=1, var_list=None):
        """
        配置trainer
        :param train_op: 优化节点

        :param var_list: 需要保存的变量名
        :param max_checkpoints: 保存模型文件最大数量
        :return:
        """
        if var_list is None:
            var_list = get_save_vars()
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=max_checkpoints)
        self.train_op = train_op

        self.compiled = True

    def init_iterator(self, set_type):
        if set_type == 'dev':
            self.session.run(self.dev_iterator.initializer)
        elif set_type == 'test':
            self.session.run(self.test_iterator.initializer)

    def backward(self):
        # 梯度累积方法
        if self.backward_op is None:
            raise ValueError("backward op is None")
        _, loss = self.session.run([self.backward_op, self.loss],
                                   feed_dict={self.is_training: True, self.handle: self.train_handle})
        return loss

    def zero_grad(self):
        # 梯度清零
        if self.zero_grad_op is None:
            raise ValueError("zero grad op is None")
        self.session.run(self.zero_grad_op)

    def train_step(self):
        """
        训练一步
        """
        self.session.run(self.train_op)
        self.global_step += 1

    def eval_step(self):
        """
        验证一步
        :return: loss + 配置的outputs
        """
        outputs = self.session.run([self.loss] + self.outputs,
                                   feed_dict={self.is_training: False, self.handle: self.dev_handle})
        return outputs

    def test_step(self):
        """
                预测一步
                :return: 配置的outputs
                """
        outputs = self.session.run(self.outputs,
                                   feed_dict={self.is_training: False, self.handle: self.test_handle})
        return outputs