# -*- coding:utf-8 -*-
# @FileName  :trainer.py
# @Time      :2021/1/31 15:24
# @Author    :huanghui
import os
import numpy as np
import tensorflow.compat.v1 as tf
from . import utils
from tqdm import tqdm
from .optimization import create_train_op
from .serving import save_pb
import importlib

_trt_available = importlib.util.find_spec("tensorflow.python.compiler.tensorrt") is not None

if _trt_available:
    from tensorflow.python.compiler.tensorrt.trt_convert import TrtGraphConverter


class Trainer:
    model_name = "model.ckpt"
    trt_name = 'model.pb'

    def __init__(self,
                 input_types=None,
                 input_shapes=None,
                 use_xla=False,
                 optimizer=None,
                 mixed_precision=False,
                 single_device=False,
                 logging=True):
        """
        tensorflow 模型多卡训练器，
        如需使用多卡，需要设置好环境变量CUDA_VISIBLE_DEVICES=0,1,2,3 (卡由自己定义，这里表示使用0 1 2 3 四块卡)
        默认使用设备的所有卡训练
        用法：
            训练： 1、创建一个trainer，如果需要使用混合精度计算，则需要在trainer创建时就传入optimizer
                    因为内部多卡计算梯度时，需要使用optimizer计算梯度，如果不使用混合精度计算，也可以在trainer
                    的compile阶段再传入optimizer
                  2、创建dataset，使用trainer.prepare_dataset方法传入创建的dataset，mode设置为train、dev、test
                     这样在train_step、eval_step、test_step调用时会选择使用相应dataset进行计算
                  3、传入model_fn, trainer.build_model(model_fn)，详情见build_model注释
                  4、调用trainer.compile配置优化节点，需要传入梯度累积步数等参数
                  5、调用trainer.compile()，训练阶段需要将trainer_op传入。
                  6、如果加载预训练参数，调用trainer.from_pretrained()；如果不加载，调用trainer.init_variables()
                  7、
                  使用trainer.train_step()优化参数，返回loss
                 验证调用trainer.eval_step()，会返回compile时的outputs和loss字段
                 测试调用trainer.test_step()，会返回outputs中的字段
                 验证和预测记得先使用trainer.init_iterator()初始化

            预测：1、创建一个trainer对象
                 2、初始化一个模型，具体件trainer的build_model方法
                 3、传入model_fn, trainer.build_model(model_fn)，详情见build_model注释
                 4、调用trainer.prepare_dataset()，将预测dataset传入
                 5、调用trainer.from_pretrained()加载参数
                 6、开始使用trainer.test_step()预测，返回的是model_fn返回的outputs字段
                   当然可以直接使用trainer.predict()预测，详情见该方法注释
        :param input_types: 定义模型输入的tensor类型，字典形式，value是tf.int32等类型定义
        :param input_shapes: 定义模型输入的tensor形状，字典形式，value是tf.TensorShape对象
        :param use_xla: 是否使用xla加速
        :param optimizer: 优化器，使用混合精度训练时需要在trainer初始化阶段就传入
        :param mixed_precision: 是否使用混合精度训练，bool变量。tf下本人测试只有开启xla的情况下混合精度才有提速效果，而且刚开始准备时间很长
        :param single_device: 是否只使用一个卡训练
        :param logging:
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

        # mode 控制接入的是训练、验证或测试dataset
        self.mode = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(self.mode, input_types, input_shapes)
        inputs = iterator.get_next()

        # 分发 dataset
        self.inputs = self.distribute_dataset(inputs)

        sess_conf = tf.ConfigProto()
        if use_xla:
            sess_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            if mixed_precision:
                tf.enable_resource_variables()
        sess_conf.gpu_options.allow_growth = True
        sess_conf.allow_soft_placement = True
        self.inited = False
        self.session = tf.Session(config=sess_conf)
        self.saver = None

        self.mode_keys = {
            'train': None, 'dev': None, 'test': None
        }

        self.iterator = {
            'train': None, 'dev': None, 'test': None
        }

        self.compiled = False
        self.optimizer = optimizer
        self.mixed_precision = mixed_precision
        self.global_step = 0  # 全局步数
        self._last_step = 0  # 上一次优化的全局步数
        self.global_step_changed = False  # 标识优化步数是否变换，避免梯度累积时重复验证的情况
        self.fgm_restore_op = None

    def is_gpu_available(self):
        with self.session:
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
        :param model_dir_or_file:
        :return:
        """
        ckpt = self.find_ckpt_file(model_dir_or_file)
        utils.init_checkpoints(ckpt, True)
        self.session.run(tf.global_variables_initializer())
        self.inited = True
        tf.logging.info("  Load model from {}".format(ckpt))

    def init_variables(self):
        self.session.run(tf.global_variables_initializer())
        self.inited = True
        tf.logging.info("  Inited global variables.")

    def save_pretrained(self, save_path_or_name):
        ckpt = self.check_file(save_path_or_name)
        self.saver.save(self.session, ckpt, self.global_step)
        tf.logging.info("  Saved model to {}".format(ckpt))

    def prepare_dataset(
            self, dataset, mode='train'):
        """
        准备模型dataset，mode可传人train、dev、test，
        传入mode后会将dataset的生成器绑定在该mode对应的string_handle，
        训练、验证、测试会根据mode调用相应数据集进行操作
        :param dataset:
        :param mode:
        :return:
        """
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
            self, model_fn, only_test=False,
            use_fgm=False,
            layer_name='word_embeddings'):
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
        :param use_fgm:  是否使用fgm对抗训练
        :param layer_name:
        :return:
        """

        if self.optimizer is None and self.mixed_precision:
            tf.logging.warn(
                "The optimizer has not been created, "
                "we will use tf.gradient to compute the gradient."
                "If you want to use mixed precision training, "
                "please configure the optimizer like tfbert/optimization/create_optimizer.")

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
                model_output = model_fn(inputs, True)

                if 'loss' in model_output:
                    if not only_test:
                        grads_and_vars = utils.compute_gradients(
                            model_output['loss'], self.optimizer)

                        # 对抗训练
                        if use_fgm:
                            grads_and_vars = utils.fgm(
                                model_fn, inputs, model_output['loss'], grads_and_vars=grads_and_vars,
                                optimizer=self.optimizer, layer_name=layer_name
                            )

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

    def compile(
            self,
            gradient_accumulation_steps=1,
            max_grad=1.0,
            max_checkpoints=1,
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
        if self.optimizer is None and optimizer is not None:
            self.optimizer = optimizer
        elif self.optimizer is not None and optimizer is not None:
            tf.logging.warn("The optimizer has been created, we will overwrite it with a new optimizer ")
            self.optimizer = optimizer
        elif self.optimizer is None and optimizer is None:
            raise ValueError("The optimizer has not been created, please prepare in the configured optimizer")

        if var_list is None:
            var_list = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=max_checkpoints)

        self.train_op = create_train_op(
            self.optimizer,
            grads_and_vars=self.grads_and_vars,
            max_grad=max_grad,
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        self.compiled = True

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
        训练一步
        """
        assert self.train_op is not None and self.compiled

        fetches = (self.train_op, tf.train.get_or_create_global_step(), self.train_outputs)
        # 使用keras全局变量指定模式，操作dropout等api。
        outputs = self.session.run(fetches,
                                   feed_dict={self.mode: self.mode_keys['train'], tf.keras.backend.learning_phase(): 1})
        loss = outputs[-1]['loss']

        global_step = outputs[1]

        if global_step > self._last_step:
            self._last_step += 1
            self.global_step += 1  # tensorflow全局步数从0开始，要加个1
            self.global_step_changed = True
        else:
            self.global_step_changed = False

        if global_step == 0 and self.global_step == 0:
            self.global_step += 1
            self.global_step_changed = True

        return loss

    def eval_step(self):
        """
        验证一步
        :return: loss + 配置的outputs
        """
        outputs = self.session.run(self.eval_outputs,
                                   feed_dict={self.mode: self.mode_keys['dev'], tf.keras.backend.learning_phase(): 0})
        return outputs

    def test_step(self):
        """
                预测一步
                :return: 配置的outputs
                """
        outputs = self.session.run(self.test_outputs,
                                   feed_dict={self.mode: self.mode_keys['test'], tf.keras.backend.learning_phase(): 0})
        return outputs

    def predict(self, set_type, output_names, total_steps=None, dataset=None):
        """
        预测的简易接口，若是自己有需求，可以调用test_step()自行写预测代码
        :param set_type: 预测模式，dev、test，输入哪个调用哪个数据集预测
        :param output_names: 返回的字段类型，字段定义需要和构建模型时定义的outputs键值一样
        :param total_steps: 可传入，dataset中包含的数据batch数量，以便于打印进度条
        :return: 字典，键值为output_names
        """
        if dataset is not None:
            self.prepare_dataset(dataset, set_type)
        outputs = {}
        for output_name in output_names:
            if output_name not in self.test_outputs or output_name not in self.eval_outputs:
                tf.logging.warn(
                    f"{output_name} is not defined when building the model, "
                    f"the returned output will not contain {output_name}")
            else:
                outputs[output_name] = None
        assert len(outputs) > 0
        self.init_iterator(set_type)
        run_fn = self.eval_step if set_type == 'dev' else self.test_step
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
        return outputs

    def export(self, serving_fn, model_dir_or_file, export_path):
        """
        模型导出为pb格式，可供tensorflow serving调用
        :param serving_fn:

        def get_serving_fn(config, args):
            def serving_fn():
                input_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='input_ids')
                input_mask = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='input_mask')
                token_type_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='token_type_ids')
                model = SequenceClassification(
                    model_type=args.model_type, config=config,
                    num_classes=len(args.labels), is_training=False,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids
                )
                inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'token_type_ids': token_type_ids}
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
