# -*- coding:utf-8 -*-
# @FileName  :trainer.py
# @Time      :2021/1/31 15:24
# @Author    :huanghui
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
from . import utils, Dataset
from .optimization import create_train_op, create_optimizer
from .serving import save_pb
import importlib
from typing import Union, Optional, Dict
from tqdm import trange, tqdm
from .utils import ProgressBar, check_dir
from . import adversarial
import platform

if platform.system() == 'Windows':
    bar_fn = ProgressBar  # win10下我使用tqdm老换行，所以自己写了一个
else:
    bar_fn = tqdm  # linux就用tqdm

_trt_available = importlib.util.find_spec("tensorflow.python.compiler.tensorrt") is not None

if _trt_available:
    from tensorflow.python.compiler.tensorrt.trt_convert import TrtGraphConverter


class BaseTrainer:
    model_name = "model.ckpt"
    trt_name = 'model.pb'

    def __init__(
            self,
            use_xla=False,
            optimizer=None,
            mixed_precision=False,
            single_device=False,
            optimizer_type='adamw',
            learning_rate=5e-5,
            num_train_epochs=1,
            train_steps=0,
            num_warmup_steps=0,
            warmup_proportion=0.,
            gradient_accumulation_steps=1,
            max_checkpoints=1,
            max_grad=1.0,
            decay_method='poly',
            logging=True):
        """
        trainer基类
        :param use_xla: 是否使用xla优化
        :param optimizer: 自定义优化器，若是不传入，需要定义下方的优化器参数
        :param optimizer_type: 优化器类型，目前支持 tfbert.optimization.create_optimizer内部的优化器
        :param learning_rate: 学习率
        :param num_train_epochs: 训练轮次
        :param train_steps: 每一轮训练步数
        :param gradient_accumulation_steps: 梯度累积步数
        :param max_checkpoints: 最大保持的ckpt数量
        :param max_grad: 最大梯度，超过进行裁剪
        :param warmup_proportion: warmup比例
        :param num_warmup_steps: warmup步数，如果传入了warmup_proportion，就不需要传了
        :param decay_method: 学习率衰减方法，见 tfbert.optimization.create_optimizer方法
        :param mixed_precision: 是否使用混合精度
        :param single_device: 是否只使用一个卡，否则使用全部卡
        :param logging: 是否显示 tf logging日志
        """
        utils.setup_xla_flags()
        if logging:
            tf.logging.set_verbosity(tf.logging.INFO)

        # 获取环境变量的devices
        self.devices = utils.devices()
        if single_device:
            self.devices = [self.devices[0]]

        # 优化节点
        self.train_op = None

        self.grads_and_vars = None

        self.train_outputs = {}
        self.eval_outputs = {}
        self.test_outputs = {}

        sess_conf = tf.ConfigProto()
        if use_xla:
            sess_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            if mixed_precision:
                tf.enable_resource_variables()
        sess_conf.gpu_options.allow_growth = True
        sess_conf.allow_soft_placement = True

        self.session = tf.Session(config=sess_conf)
        self.saver = None
        self.inited = False
        self.compiled = False
        self.finished_build = False

        self.num_train_epochs = num_train_epochs
        self.max_checkpoints = max_checkpoints
        self.max_grad = max_grad
        self.num_train_steps = (train_steps * num_train_epochs // gradient_accumulation_steps)
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.warmup_proportion = warmup_proportion
        if warmup_proportion > 0:
            self.num_warmup_steps = self.num_train_steps * warmup_proportion

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.decay_method = None if self.num_train_steps == 0 else decay_method
        self.optimizer_type = optimizer_type
        self.optimizer = optimizer
        self.mixed_precision = mixed_precision

        self.global_step = 0  # 全局步数
        self.forward_steps = 0  # 前向步数
        self.global_step_changed = False  # 标识优化步数是否变换，避免梯度累积时重复验证的情况

    def check_init(self):
        if not self.inited:
            raise ValueError("please init variables(init_variables) or load pretrained variables (from_pretrained)!")

    def check_build(self):
        if not self.finished_build:
            raise ValueError("please build your model !")

    def check_compile(self):
        if not self.compiled:
            raise ValueError("please build train op by using trainer.compile")

    def is_gpu_available(self):
        with self.session:
            return tf.test.is_gpu_available()

    @property
    def num_params(self):
        self.check_build()
        # 参数量，需要配置好图、会话以后用
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        return num_params

    @property
    def num_devices(self):
        # devices 数量
        return len(self.devices)

    def check_file(self, filename_or_path):
        if os.path.isdir(filename_or_path):
            ckpt = os.path.join(filename_or_path, self.model_name)
        else:
            ckpt = filename_or_path
        return ckpt

    def find_ckpt_file(self, model_dir_or_file):
        if os.path.isdir(model_dir_or_file):
            ckpt = tf.train.latest_checkpoint(model_dir_or_file)
            if ckpt is None:
                ckpt = os.path.join(model_dir_or_file, self.model_name)
        else:
            ckpt = model_dir_or_file
        return ckpt

    def from_pretrained(self, model_dir_or_file):
        """
        加载模型参数
        :param model_dir_or_file:
        :return:
        """
        self.check_build()
        ckpt = self.find_ckpt_file(model_dir_or_file)
        utils.init_checkpoints(ckpt, True)
        self.session.run(tf.global_variables_initializer())
        self.inited = True
        tf.logging.info("  Load model from {}".format(ckpt))

    def init_variables(self):
        """
        初始化参数
        :return:
        """
        self.check_build()
        self.session.run(tf.global_variables_initializer())
        self.inited = True
        tf.logging.info("  Inited global variables.")

    def save_pretrained(self, save_path_or_name, add_global_step=True):
        self.check_build()
        ckpt = self.check_file(save_path_or_name)
        self.saver.save(self.session, ckpt, self.global_step if add_global_step else None)
        tf.logging.info("  Saved model to {}".format(ckpt))
        return ckpt

    def prepare_optimizer(self):
        """
        默认创建的优化器
        :return:
        """
        self.optimizer = create_optimizer(
            learning_rate=self.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            optimizer_type=self.optimizer_type,
            decay_method=self.decay_method,
            mixed_precision=self.mixed_precision,
        )

    def compile(
            self,
            gradient_accumulation_steps=None,
            max_grad=None,
            max_checkpoints=None,
            optimizer=None,
            var_list=None):
        """
        配置优化器
        :param gradient_accumulation_steps: 梯度累积步数
        :param max_grad: 最大梯度，大于这个梯度都会被裁剪成这个值
        :param max_checkpoints:
        :param optimizer: 如果在trainer初始化就已经传入，就不需要再传了
        :param var_list:
        :return:
        """
        self.check_build()
        self.optimizer = optimizer
        if self.optimizer is None:
            self.prepare_optimizer()
        if gradient_accumulation_steps is not None:
            self.gradient_accumulation_steps = gradient_accumulation_steps
        if max_checkpoints is not None:
            self.max_checkpoints = max_checkpoints
        if max_grad is not None:
            self.max_grad = max_grad

        if var_list is None:
            var_list = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.max_checkpoints)

        self.train_op = create_train_op(
            self.optimizer,
            grads_and_vars=self.grads_and_vars,
            max_grad=self.max_grad,
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )

        self.compiled = True

    def _train_step(self, feed_dict):
        self.check_compile()
        fetches = (self.train_op, self.train_outputs)
        feed_dict[tf.keras.backend.learning_phase()] = 1
        # 使用keras全局变量指定模式，操作dropout等api。
        outputs = self.session.run(fetches, feed_dict=feed_dict)
        loss = outputs[-1]['loss']

        self.forward_steps += 1
        if self.forward_steps % self.gradient_accumulation_steps == 0:
            self.global_step += 1

        return loss

    def _eval_step(self, feed_dict):
        self.check_init()
        feed_dict[tf.keras.backend.learning_phase()] = 0
        outputs = self.session.run(self.eval_outputs, feed_dict=feed_dict)
        return outputs

    def _predict_step(self, feed_dict):
        self.check_init()
        feed_dict[tf.keras.backend.learning_phase()] = 0
        outputs = self.session.run(self.test_outputs, feed_dict=feed_dict)
        return outputs

    def export(self, serving_fn, model_dir_or_file, export_path):
        """
        模型导出为pb格式，可供tensorflow serving调用
        :param serving_fn:

        def get_serving_fn(config, args):
            def serving_fn():
                input_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='input_ids')
                attention_mask = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='attention_mask')
                token_type_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='token_type_ids')
                model = SequenceClassification(
                    model_type=args.model_type, config=config,
                    num_classes=len(args.labels), is_training=False,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
                outputs = {'logits': model.logits}
                return inputs, outputs
            return serving_fn  # 返回这个就是serving fn
        :param model_dir_or_file: 训练保存的模型权重所在文件夹和文件名
        :param export_path: pb导出地址，文件夹
        :return:
        """
        # 导出pb这里还不方便，需要重置图
        # 主要是因为之前trainer的mode是一个占位符，直接导出pb会增加一个输入，不方便
        # 最终觉得还是这样导出的结果好点
        tf.reset_default_graph()

        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        sess_conf.allow_soft_placement = True
        with tf.Session(config=sess_conf) as sess:
            inputs, outputs = serving_fn()
            utils.init_checkpoints(self.find_ckpt_file(model_dir_or_file), False)
            sess.run(tf.global_variables_initializer())
            save_pb(sess, export_path, inputs=inputs, outputs=outputs)
            tf.logging.info('export model to {}'.format(export_path))

    def export_to_tf_trt(
            self, serving_fn, model_dir_or_file,
            export_path, output_node_names=None):
        """
        没有安装好tensorRT的GPU，因此还没测试过这个功能

        模型导出为tf tensorRT格式，具体serving fn和上面export一致
        :param serving_fn:
        :param model_dir_or_file:
        :param export_path:
        :param output_node_names: tensorRT模型的输出节点
        :return:
        """
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        tf.reset_default_graph()

        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        sess_conf.allow_soft_placement = True
        with tf.Session(config=sess_conf) as sess:
            inputs, outputs = serving_fn()
            if output_node_names is None:
                output_node_names = []
                for k, v in outputs.items():
                    output_node_names.append(v.name.split(':')[0])
            utils.init_checkpoints(self.find_ckpt_file(model_dir_or_file), False)
            sess.run(tf.global_variables_initializer())
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(), output_node_names)

            num_nodes = len(frozen_graph.node)
            tf.logging.info('Converting graph using TensorFlow-TensorRT...')

            converter = TrtGraphConverter(
                input_graph_def=frozen_graph,
                nodes_blacklist=output_node_names,
                max_workspace_size_bytes=(4096 << 20) - 1000,
                precision_mode="FP16" if self.mixed_precision else "FP32",
                minimum_segment_size=len(output_node_names),
                is_dynamic_op=True,
                maximum_cached_engines=1000
            )
            frozen_graph = converter.convert()

            tf.logging.info('Total node count before and after TF-TRT conversion:',
                            num_nodes, '->', len(frozen_graph.node))
            tf.logging.info('TRT node count:',
                            len([1 for n in frozen_graph.node if str(n.op) == 'TRTEngineOp']))

            with tf.io.gfile.GFile(os.path.join(export_path, self.trt_name), "wb") as f:
                f.write(frozen_graph.SerializeToString())
        return frozen_graph, output_node_names


class Trainer(BaseTrainer):
    def __init__(self,
                 train_dataset: Optional[Union[tf.data.Dataset, Dataset]] = None,
                 eval_dataset: Optional[Union[tf.data.Dataset, Dataset]] = None,
                 output_types=None,
                 output_shapes=None,
                 metric_fn=None,
                 post_process_fn=None,
                 use_xla=False,
                 optimizer=None,
                 optimizer_type='adamw',
                 learning_rate=5e-5,
                 num_train_epochs=1,
                 train_steps=0,
                 eval_steps=0,
                 gradient_accumulation_steps=1,
                 max_checkpoints=1,
                 max_grad=1.0,
                 warmup_proportion=0.,
                 num_warmup_steps=0,
                 decay_method='poly',
                 mixed_precision=False,
                 single_device=False,
                 logging=True):
        """
        tensorflow训练trainer封装，采用tf dataset进行数据喂入，使用string handle动态控制dataset

        先初始化，然后调用build_model传入model_fn构建模型；
        训练的话需要接着调用compile配置优化op；
        然后使用from_pretrained或者init_variables进行参数初始化；
        之后可以train()进行训练，evaluate()进行验证，predict()进行预测；
        :param train_dataset: 训练dataset，可以为tf.data.Dataset，也可以为我自定义的Dataset类型，定义代码在 tfbert.data.dataset.Dataset
        :param eval_dataset: 验证dataset，同 train_dataset
        :param output_types: 输入数据类型定义，可以使用我自定义的Dataset直接获取，方法有output_types_and_shapes和get_output_types_and_shapes，具体看相应注释
        :param output_shapes: 输入数据的shape定义，同上。这俩在train_dataset或者dev_dataset传入的时候可以不传
        :param metric_fn: 评估函数，该函数接收的输入为post_process_fn的输出，若是post_process_fn不为None的话，如果为None就是trainer.predict方法的结果
        :param post_process_fn: 结果后处理函数，该方法接收输入为trainer.predict方法的结果，不需处理的话可以不传入
        :param use_xla: 是否使用xla优化
        :param optimizer: 自定义优化器，若是不传入，需要定义下方的优化器参数
        :param optimizer_type: 优化器类型，目前支持 tfbert.optimization.create_optimizer内部的优化器
        :param learning_rate: 学习率
        :param num_train_epochs: 训练轮次
        :param train_steps: 每一轮训练步数
        :param eval_steps: 验证一轮的步数
        :param gradient_accumulation_steps: 梯度累积步数
        :param max_checkpoints: 最大保持的ckpt数量
        :param max_grad: 最大梯度，超过进行裁剪
        :param warmup_proportion: warmup比例
        :param num_warmup_steps: warmup步数，如果传入了warmup_proportion，就不需要传了
        :param decay_method: 学习率衰减方法，见 tfbert.optimization.create_optimizer方法
        :param mixed_precision: 是否使用混合精度
        :param single_device: 是否只使用一个卡，否则使用全部卡
        :param logging: 是否显示 tf logging日志
        """
        super(Trainer, self).__init__(
            use_xla,
            optimizer,
            mixed_precision,
            single_device,
            optimizer_type,
            learning_rate,
            num_train_epochs,
            train_steps,
            num_warmup_steps,
            warmup_proportion,
            gradient_accumulation_steps,
            max_checkpoints,
            max_grad,
            decay_method,
            logging
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        if (self.train_dataset or self.eval_dataset) is None:
            if output_shapes is None and output_types is None:
                raise ValueError(
                    f"If you train_dataset and eval_dataset are both None, output_types and output_shapes can not be None")
        else:
            if isinstance(self.train_dataset or self.eval_dataset, Dataset):
                output_types, output_shapes = (self.train_dataset or self.eval_dataset).output_types_and_shapes()
            else:
                output_types, output_shapes = Dataset.get_output_types_and_shapes(
                    (self.train_dataset or self.eval_dataset), use_none=True)

        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if isinstance(self.train_dataset, Dataset):
            self.train_steps = len(train_dataset)
            self.num_train_steps = (self.train_steps * num_train_epochs // gradient_accumulation_steps)
            if warmup_proportion > 0:
                self.num_warmup_steps = self.num_train_steps * warmup_proportion
        self.eval_steps = eval_steps
        if isinstance(self.eval_dataset, Dataset):
            self.eval_steps = len(self.eval_dataset)

        # mode 控制接入的是训练、验证或测试dataset
        self.mode = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(self.mode, output_types, output_shapes)
        inputs = iterator.get_next()

        # 分发 dataset
        self.inputs = self.distribute_dataset(inputs)

        self.mode_keys = {
            'train': None, 'dev': None, 'test': None
        }

        self.iterator = {
            'train': None, 'dev': None, 'test': None
        }
        self.metric_fn = metric_fn
        self.post_process_fn = post_process_fn

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

    def prepare_dataset(
            self, dataset: Union[tf.data.Dataset, Dataset], mode='train'):
        """
        准备模型dataset，mode可传人train、dev、test，
        传入mode后会将dataset的生成器绑定在该mode对应的string_handle，
        训练、验证、测试会根据mode调用相应数据集进行操作
        :param dataset:
        :param mode:
        :return:
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.format_to_tf_dataset()
        if mode not in self.mode_keys:
            raise ValueError("mode must be {}".format("、".join(self.mode_keys.keys())))
        if mode == 'train':
            iterator = dataset.make_one_shot_iterator()
        else:
            iterator = dataset.make_initializable_iterator()
        handle = self.session.run(iterator.string_handle())
        self.mode_keys[mode] = handle
        self.iterator[mode] = iterator

    def build_model(
            self,
            model_fn,
            only_test=False,
            adversarial_type=None,
            **kwargs
    ):
        """
        传入model_fn，也就是 model 构造函数，model_fn只能接收inputs和is_training

        model_fn 返回一个字典结果，训练需要给定loss
        如果需要验证和测试，需要传入outputs，outputs是一个字典
        examples:

        def get_model_fn(config, args):
            def model_fn(inputs, is_training):
                model = SequenceClassification(
                    model_type=args.model_type, config=config,
                    num_classes=len(args.labels), is_training=is_training,
                    **inputs)
                loss = model.loss / args.gradient_accumulation_steps
                return {
                    'loss': loss,
                    'outputs': {'logits': model.logits, 'label_ids': inputs['label_ids']}}

            return model_fn

        trainer.build_model(get_model_fn(model_type, config, num_classes=len(labels)))

        :param model_fn:
        :param only_test: 是否只测试，True的话梯度这些就不计算了
        :param adversarial_type: 对抗训练类型
        :return:
        """

        if self.optimizer is None and self.mixed_precision:
            tf.logging.warn(
                "you want to use mixed precision training and the optimizer has not been created, "
                "we will create a optimizer.")
            # 混合精度需要使用开启fp16的优化器计算梯度，因此不能使用tf.gradient
            self.prepare_optimizer()

        adversarial_params = {
            'fgm': {'layer_name': 'word_embeddings', 'epsilon': 0.5},
            'pgd': {'layer_name': 'word_embeddings', 'epsilon': 0.05, 'n_loop': 2},
            'freelb': {'layer_name': 'word_embeddings', 'epsilon': 0.3, 'n_loop': 3}
        }
        if adversarial_type is not None:
            assert adversarial_type in adversarial_params

            for k, v in kwargs.items():
                if k in adversarial_params[adversarial_type]:
                    adversarial_params[k] = v

            if adversarial_type == 'freelb':
                adversarial_params[adversarial_type]['batch_size'] = kwargs['batch_size']
                adversarial_params[adversarial_type]['max_length'] = kwargs['max_length']

        tower_losses, tower_grads_and_vars = [], []
        outputs = {}
        for i, device in enumerate(self.devices):
            reuse = True if i > 0 else None
            with tf.device(device), \
                    tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                # 这种做法还是不能判断，tensorflow里边需要tf.cond创建if
                # is_training = (self.mode == self.mode_keys['train'])

                # 直接默认都是训练模式，然后使用tf.keras.backend.learning_phase()控制dropout等layer
                inputs = self.get_inputs(device)
                if adversarial_type == 'fgm':
                    model_output = adversarial.fgm(
                        model_fn, inputs, optimizer=self.optimizer,
                        layer_name=adversarial_params['fgm']['layer_name'],
                        epsilon=adversarial_params['fgm']['epsilon'])

                elif adversarial_type == 'pgd':
                    model_output = adversarial.pgd(
                        model_fn, inputs, optimizer=self.optimizer,
                        layer_name=adversarial_params['pgd']['layer_name'],
                        epsilon=adversarial_params['pgd']['epsilon'],
                        n_loop=adversarial_params['pgd']['n_loop']
                    )
                elif adversarial_type == 'freelb':
                    model_output = adversarial.freelb(
                        model_fn, inputs, adversarial_params['freelb']['batch_size'],
                        adversarial_params['freelb']['max_length'],
                        optimizer=self.optimizer,
                        layer_name=adversarial_params['freelb']['layer_name'],
                        epsilon=adversarial_params['freelb']['epsilon'],
                        n_loop=adversarial_params['freelb']['n_loop']
                    )
                else:
                    model_output = model_fn(inputs, True)

                if isinstance(model_output, adversarial.AdversarialOutput):
                    grads_and_vars = model_output.grads_and_vars
                    tower_grads_and_vars.append(grads_and_vars)

                    model_output = model_output.outputs

                    if 'loss' in model_output:
                        tower_losses.append(model_output['loss'])

                elif 'loss' in model_output:
                    if not only_test:
                        grads_and_vars = utils.compute_gradients(
                            model_output['loss'], self.optimizer)

                        tower_grads_and_vars.append(grads_and_vars)
                    tower_losses.append(model_output['loss'])
                if 'outputs' in model_output:
                    for k, v in model_output['outputs'].items():
                        if k not in outputs:
                            outputs[k] = []
                        outputs[k].append(v)

        loss = None
        grads_and_vars = None
        if len(tower_losses) > 0:
            loss = (tf.add_n(tower_losses) / len(tower_losses)) if self.num_devices > 1 else tower_losses[0]
            if tower_grads_and_vars:
                grads_and_vars = utils.average_grads_and_vars(tower_grads_and_vars) if self.num_devices > 1 else \
                    tower_grads_and_vars[0]

        if len(outputs) > 0:
            for k, v in outputs.items():
                outputs[k] = tf.concat(v, 0) if self.num_devices > 1 else v[0]

        self.train_outputs['loss'] = loss  # 训练只返回loss，若是有需求，可自行更改

        self.eval_outputs = {'loss': loss}  # 验证返回loss，和定义的outputs
        self.eval_outputs.update(outputs)

        self.test_outputs = outputs  # 测试返回outputs

        self.grads_and_vars = grads_and_vars
        self.finished_build = True

    def init_iterator(self, mode):
        """
        初始化dataset的生成器，在验证和测试之前调用
        :param mode:
        :return:
        """
        if mode not in ['dev', 'test']:
            raise ValueError("mode must be dev、test. If you want to initialize train iterator, you should do nothing")

        if self.iterator[mode] is None:
            raise ValueError(f"Please prepare {mode} dataset before initialize")

        self.session.run(self.iterator[mode].initializer)

    def train_step(self):
        """
        训练一步，调用train_dataset进行训练，运行op为train op，返回loss
        """
        feed_dict = {self.mode: self.mode_keys['train']}
        return self._train_step(feed_dict)

    def eval_step(self):
        """
        验证一步，调用dev_dataset验证，运行op为trainer的dev_outputs
        :return: loss + 配置的outputs
        """
        feed_dict = {self.mode: self.mode_keys['dev']}
        return self._eval_step(feed_dict)

    def predict_step(self):
        """
                预测一步,调用test_dataset验证，运行op为trainer的 test_outputs
                :return: 配置的outputs
                """
        feed_dict = {self.mode: self.mode_keys['test']}
        return self._predict_step(feed_dict)

    def check_dataset_and_steps(self, set_type, dataset, steps):
        assert set_type in ['dev', 'train']
        self_dataset = self.eval_dataset if set_type == 'dev' else self.train_dataset
        if isinstance(dataset, Dataset):
            steps = dataset.num_batch
        elif isinstance(self_dataset, Dataset):
            steps = self_dataset.num_batch

        if set_type == 'dev' and steps > 0:
            self.eval_steps = steps

        if set_type == 'train' and steps > 0:
            self.train_steps = steps
            self.num_train_steps = (self.train_steps * self.num_train_epochs // self.gradient_accumulation_steps)
            if self.warmup_proportion > 0:
                self.num_warmup_steps = self.num_train_steps * self.warmup_proportion

        if set_type == 'train' and self.train_steps <= 0:
            raise ValueError(
                "Train steps can not be None if you want to train. Maybe you prapre a dataset whoose class is Dataset")

        if dataset is not None:
            self.prepare_dataset(dataset, mode=set_type)
        if self_dataset is not None and self.iterator[set_type] is None:
            self.prepare_dataset(self_dataset, mode=set_type)
        if self.iterator[set_type] is None:
            raise ValueError(f"set_type: {set_type} should be prepared.")
        return steps

    def evaluate(
            self,
            eval_dataset: Optional[Union[tf.data.Dataset, Dataset]] = None,
            eval_steps=0, metric_fn=None,
            post_process_fn=None):
        """
        验证接口
        :param eval_dataset:
        :param eval_steps: 验证一轮的步数
        :param metric_fn: 评估调用方法，将接受post_process_fn的输出或者已定义的model fn的 outputs
        :param post_process_fn: 对model fn输出的outputs后处理方法
        :return:
        """
        if metric_fn is None and self.metric_fn is None:
            raise ValueError("Please pass in the evaluation function (metric_fn)!")
        elif self.metric_fn is not None:
            metric_fn = self.metric_fn

        self.check_init()
        self.check_dataset_and_steps('dev', eval_dataset, eval_steps)
        outputs = self.predict(
            'dev', list(self.eval_outputs.keys()), self.eval_steps,
            post_process_fn=post_process_fn)
        metric = metric_fn(outputs)
        return metric

    def train(self,
              train_dataset: Optional[Union[tf.data.Dataset, Dataset]] = None,
              train_steps=0,
              output_dir=None,
              evaluate_during_training=False,
              metric_fn=None,
              post_process_fn=None,
              logging_steps=0,
              saving_steps=0,
              greater_is_better=True,
              metric_for_best_model=None,
              load_best_model=True,
              eval_dataset: Optional[Union[tf.data.Dataset, Dataset]] = None,
              eval_steps=0):
        """
        训练接口
        :param train_dataset:
        :param train_steps: 这个是一轮的步数
        :param output_dir:
        :param evaluate_during_training:
        :param metric_fn: 评估方法
        :param post_process_fn: predict的outputs 后处理方法
        :param logging_steps:
        :param saving_steps:
        :param greater_is_better: 是否验证结果高表示效果好
        :param metric_for_best_model: 评估模型指标字段，也就是用metric fn返回字典结果中哪个结果筛选模型
        :param load_best_model: 是否在训练结束后加载保存的最优模型
        :param eval_dataset:
        :param eval_steps: 验证一轮的步数
        :return:
        """
        self.check_compile()
        self.check_init()
        self.check_dataset_and_steps('train', train_dataset, train_steps)
        if eval_dataset is not None:
            self.check_dataset_and_steps('dev', eval_dataset, eval_steps)
        if evaluate_during_training:
            if metric_fn is None and self.metric_fn is None:
                raise ValueError("Please pass in the evaluation function (metric_fn)!")
            if post_process_fn is None and self.post_process_fn is None:
                tf.logging.warn("post_process_fn is None, we will not process the prediction result")
            if logging_steps <= 0:
                raise ValueError(
                    "If you need to verify while training, please ensure that the logging steps are greater than 0")
            if metric_for_best_model is None:
                raise ValueError(
                    "If you need to verify while training, please provide the evaluation field (metric_for_best_model)")
        if output_dir is None:
            tf.logging.info("output_dir is None, we will save files to 'output' by default")
            output_dir = "output"
        check_dir(output_dir)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num epochs = {}".format(self.num_train_epochs))
        tf.logging.info("  optimizer steps = %d", self.num_train_steps)
        tf.logging.info("  gradient accumulation steps = %d", self.gradient_accumulation_steps)
        tf.logging.info("  Num devices = {}".format(self.num_devices))
        tf.logging.info("  Num params = {}".format(self.num_params))

        report = {}
        never_saved = True
        best_score = 0 if greater_is_better else 1e8
        best_model_file = None
        for epoch in trange(self.num_train_epochs):
            epoch_iter = bar_fn(range(self.train_steps), desc='epoch {}'.format(epoch + 1))
            train_losses = 0
            for step in epoch_iter:
                train_loss = self.train_step()
                train_losses += train_loss
                epoch_iter.set_postfix(loss=round(train_losses / (step + 1), 4))

                if evaluate_during_training and self.global_step > 0 and (
                        self.global_step % logging_steps == 0 or self.global_step == self.num_train_steps):
                    metric = self.evaluate(self.eval_dataset, self.eval_steps, metric_fn, post_process_fn)
                    score = metric[metric_for_best_model]
                    should_save = ((score > best_score) if greater_is_better else (score < best_score))
                    if should_save:
                        best_score = score
                        best_model_file = self.save_pretrained(
                            output_dir, add_global_step=False
                        )
                        never_saved = False
                    tf.logging.info("***** eval results *****")
                    tf.logging.info(" global step : {}".format(self.global_step))
                    tf.logging.info(" eval score : {:.4f}".format(score))
                    tf.logging.info(" best score : {:.4f}".format(best_score))
                    report[self.global_step] = metric
                if not evaluate_during_training and self.global_step > 0 and (
                        saving_steps > 0 and self.global_step % saving_steps == 0):
                    check_dir(output_dir)
                    self.save_pretrained(output_dir, add_global_step=True)
                    never_saved = False
        if never_saved:
            self.save_pretrained(output_dir, add_global_step=False)

        report['best_score'] = best_score
        json.dump(
            report, open(os.path.join(output_dir, 'train_report.json'), 'w', encoding='utf-8'),
            ensure_ascii=False, indent=4)

        tf.logging.info("***** Finished Training *****")
        if load_best_model and best_model_file is not None:
            tf.logging.info(f"Loading best model from {best_model_file} (score: {best_score}).")
            self.from_pretrained(best_model_file)
        return report

    def predict(self, set_type, output_names=None, total_steps=None,
                dataset: Optional[Union[tf.data.Dataset, Dataset]] = None,
                post_process_fn=None):
        """
        预测的简易接口，若是自己有需求，可以调用predict_step()自行写预测代码
        :param set_type: 预测模式，dev、test，输入哪个调用哪个数据集预测
        :param output_names: 返回的字段类型，字段定义需要和构建模型时定义的outputs键值一样
        :param total_steps: 可传入，dataset中包含的数据batch数量，以便于打印进度条
        :param dataset:
        :param post_process_fn: 结果后处理方法，若是传入，则将预测结果传入进行处理
        :return: 字典，键值为output_names
        """
        if isinstance(dataset, Dataset):
            total_steps = len(dataset)
        if dataset is not None:
            self.prepare_dataset(dataset, set_type)

        if post_process_fn is None and self.post_process_fn is None:
            tf.logging.warn("post_process_fn is None, we will not process the prediction result")
        elif self.post_process_fn is not None:
            post_process_fn = self.post_process_fn

        outputs = {}
        if output_names is None:
            output_names = list(self.test_outputs.keys())
        for output_name in output_names:
            if output_name not in self.test_outputs and output_name not in self.eval_outputs:
                tf.logging.warn(
                    f"{output_name} is not defined when building the model, "
                    f"the returned output will not contain {output_name}")
            else:
                outputs[output_name] = None
        assert len(outputs) > 0
        self.init_iterator(set_type)
        run_fn = self.eval_step if set_type == 'dev' else self.predict_step
        predict_iter = tqdm(
            iterable=range(total_steps) if total_steps is not None else None,
            desc="Predicting"
        )
        count = 0
        while True:
            try:
                output = run_fn()
                for k, v in output.items():
                    if k in outputs:
                        if outputs[k] is None:
                            outputs[k] = v
                        else:
                            if k == 'loss':
                                outputs[k] += v
                                count += 1
                            else:
                                outputs[k] = np.append(outputs[k], v, 0)
                predict_iter.update(1)

            except tf.errors.OutOfRangeError:
                break
        if 'loss' in outputs:
            outputs['loss'] /= count
        predict_iter.close()
        if post_process_fn is not None:
            outputs = post_process_fn(outputs)
        return outputs


class SimplerTrainer(BaseTrainer):
    """
    采用 feed 模式的trainer，相比 dataset要简单点，但是速度不够
    """

    def __init__(self,
                 use_xla=False,
                 optimizer=None,
                 mixed_precision=False,
                 single_device=False,
                 optimizer_type='adamw',
                 learning_rate=5e-5,
                 num_train_epochs=1,
                 train_steps=0,
                 gradient_accumulation_steps=1,
                 max_checkpoints=1,
                 max_grad=1.0,
                 warmup_proportion=0,
                 num_warmup_steps=0,
                 decay_method='poly',
                 logging=True):
        super(SimplerTrainer, self).__init__(
            use_xla,
            optimizer,
            mixed_precision,
            single_device,
            optimizer_type,
            learning_rate,
            num_train_epochs,
            train_steps,
            num_warmup_steps,
            warmup_proportion,
            gradient_accumulation_steps,
            max_checkpoints,
            max_grad,
            decay_method,
            logging
        )

        self.inputs = []

    def build_model(self,
                    model_fn,
                    only_test=False):
        """

        :param model_fn:
        :param only_test:
        :return:
        """
        if self.optimizer is None and self.mixed_precision:
            self.prepare_optimizer()
        tower_losses, tower_grads_and_vars = [], []
        outputs = {}
        self.inputs = []
        for i, device in enumerate(self.devices):
            reuse = True if i > 0 else None
            with tf.device(device), \
                    tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                # 直接默认都是训练模式，然后使用tf.keras.backend.learning_phase()控制dropout等layer
                model_inputs, model_outputs = model_fn()
                per_device_inputs = {}
                for k, v in model_inputs.items():
                    per_device_inputs[k] = v
                self.inputs.append(per_device_inputs)

                if 'loss' in model_outputs:
                    if not only_test:
                        grads_and_vars = utils.compute_gradients(
                            model_outputs['loss'], self.optimizer)
                        tower_grads_and_vars.append(grads_and_vars)
                    tower_losses.append(model_outputs['loss'])

                for k, v in model_outputs.items():
                    if k not in outputs:
                        outputs[k] = []
                    outputs[k].append(v)

        loss = None
        grads_and_vars = None
        if len(tower_losses) > 0:
            loss = (tf.add_n(tower_losses) / len(tower_losses)) if self.num_devices > 1 else tower_losses[0]
            if tower_grads_and_vars:
                grads_and_vars = utils.average_grads_and_vars(tower_grads_and_vars) if self.num_devices > 1 else \
                    tower_grads_and_vars[0]

        if len(outputs) > 0:
            for k, v in outputs.items():
                outputs[k] = tf.concat(v, 0) if self.num_devices > 1 else v[0]

        self.train_outputs['loss'] = loss  # 训练只返回loss，若是有需求，可自行更改

        self.eval_outputs = {'loss': loss}  # 验证返回loss，和定义的outputs
        self.eval_outputs.update(outputs)

        self.test_outputs = outputs  # 测试返回outputs

        self.grads_and_vars = grads_and_vars
        self.finished_build = True

    def feed_dict(self, inputs):
        total = len(list(inputs.values())[0])
        batch = total // self.num_devices
        feed_dict = {}
        for i in range(self.num_devices):
            start = i * batch
            end = (i + 1) * batch
            per_device_inputs = self.inputs[i]
            for k, v in inputs.items():
                if i == self.num_devices - 1:
                    feed_dict[per_device_inputs[k]] = v[start:]  # 最后一个device拿到剩余全部数据，避免不能整除的情况
                else:
                    feed_dict[per_device_inputs[k]] = v[start: end]
        return feed_dict

    def train_step(self, inputs):
        """
        训练一步
        """
        feed_dict = self.feed_dict(inputs)

        return self._train_step(feed_dict)

    def eval_step(self, inputs):
        feed_dict = self.feed_dict(inputs)
        return self._eval_step(feed_dict)

    def predict_step(self, inputs):
        feed_dict = self.feed_dict(inputs)
        return self._predict_step(feed_dict)

    def __call__(self, inputs):
        return self.predict_step(inputs)

    def predict(self, inputs, output_names=None, batch_size=8):
        """
        预测的简易接口，若是自己有需求，可以调用predict_step()自行写预测代码
        :param inputs: 字典型输入数据，字段和mode fn 的inputs对应
        :param output_names: 返回的字段类型，字段定义需要和构建模型时定义的outputs键值一样
        :param batch_size: 可传入，dataset中包含的数据batch数量，以便于打印进度条
        :return: 字典，键值为output_names
        """
        self.check_init()
        if output_names is not None:
            output_names = self.test_outputs.keys()
        outputs = {}
        for output_name in output_names:
            if output_name not in self.test_outputs or output_name not in self.eval_outputs:
                tf.logging.warn(
                    f"{output_name} is not defined when building the model, "
                    f"the returned output will not contain {output_name}")
            else:
                outputs[output_name] = None
        assert len(outputs) > 0
        pred_fn = self.eval_step if 'loss' in output_names else self.predict_step
        num_features = len(inputs[list(inputs.keys())[0]])
        total = (num_features + batch_size - 1) // batch_size
        for i in tqdm(range(total), desc='Predicting'):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch = {}
            for k, v in inputs.items():
                batch[k] = v[start: end]
            output = pred_fn(batch)
            for k, v in output.items():
                if k in outputs:
                    if outputs[k] is None:
                        outputs[k] = v
                    else:
                        if k == 'loss':
                            outputs[k] += v
                        else:
                            outputs[k] = np.append(outputs[k], v, 0)

        if 'loss' in outputs:
            outputs['loss'] /= total
        return outputs
