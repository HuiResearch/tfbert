# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: tokenization_base.py
@date: 2020/09/08
"""
from . import BaseConfig
import re
import tensorflow.compat.v1 as tf
import os
import shutil


class BertConfig(BaseConfig):
    def __init__(self,
                 vocab_size,
                 embedding_size=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class ALBertConfig(BaseConfig):

    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_size=4096,
                 num_hidden_layers=12,
                 num_hidden_groups=1,
                 num_attention_heads=64,
                 intermediate_size=16384,
                 inner_group_num=1,
                 down_scale_factor=1,
                 hidden_act="gelu",
                 hidden_dropout_prob=0,
                 attention_probs_dropout_prob=0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.inner_group_num = inner_group_num
        self.down_scale_factor = down_scale_factor
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class ElectraConfig(BaseConfig):
    """Configuration for `BertModel` (ELECTRA uses the same ptm as BERT)."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_checkpoint(cls, checkpoint_path, **kwargs):
        """
        由于electra 没有给出config.json，所以构建一个方法，从checkpoint中读取配置信息。
        :param checkpoint_path: electra模型的checkpoint文件
        :param kwargs:
        :return:
        """
        # 参数映射，checkpoint变量名: (config配置参数，配置参数属于变量shape的哪个维度的大小)
        param_map = {
            'electra/embeddings/word_embeddings': ('vocab_size', 0),
            'electra/encoder/layer_0/attention/output/dense/bias': ('hidden_size', 0),
            'electra/encoder/layer_0/intermediate/dense/bias': ('intermediate_size', 0),
            'electra/embeddings/position_embeddings': ('max_position_embeddings', 0),
            'electra/embeddings/token_type_embeddings': ('type_vocab_size', 0)
        }
        # 基本参数
        param = {
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'hidden_act': 'gelu',
            'initializer_range': 0.02
        }

        # 加载checkpoint，获取相应参数
        init_vars = tf.train.list_variables(checkpoint_path)
        num_hidden_layers = 0
        for x in init_vars:
            name, shape = x[0], x[1]
            if name in param_map:
                param[param_map[name][0]] = shape[param_map[name][1]]

            if 'layer_' in name:
                layer = re.match(".*?layer_(\\d+)/.*?", name).group(1)
                if int(layer) >= num_hidden_layers:
                    num_hidden_layers = int(layer)

        param['num_hidden_layers'] = num_hidden_layers + 1
        param['num_attention_heads'] = max(1, param["hidden_size"] // 64)

        return cls(**param, **kwargs)


class GlyceBertConfig(BaseConfig):
    def __init__(self,
                 vocab_size,
                 embedding_size=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 config_path="",
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.config_path = config_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        '''
        从文件夹或文件中加载config
        :param pretrained_model_name_or_path:
        :param kwargs:
        :return:
        '''

        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, cls.filename)
            config_path = os.path.join(pretrained_model_name_or_path, "config")
        elif os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
            dir_ = os.path.split(config_file)[0]
            config_path = os.path.join(dir_, 'config')
        else:
            raise ValueError('Config path should be a directory or file')

        config_dict = cls._dict_from_json_file(config_file)
        kwargs['config_path'] = config_path
        return cls.from_dict(config_dict, **kwargs)

    def save_pretrained(self, save_dir_or_file):
        if os.path.isdir(save_dir_or_file):
            output_config_file = os.path.join(save_dir_or_file, self.filename)
            config_path = os.path.join(save_dir_or_file, 'config')
        else:
            output_config_file = save_dir_or_file
            config_path = os.path.join(os.path.split(save_dir_or_file)[0], "config")
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        filenames = os.listdir(self.config_path)
        if len(filenames) > 0:
            for filename in filenames:
                if filename.endswith('.npy'):
                    shutil.copyfile(
                        os.path.join(self.config_path, filename), os.path.join(config_path, filename)
                    )
        self.save_to_json_file(output_config_file)
        tf.logging.info('  Configuration saved in {}'.format(output_config_file))
        return output_config_file
