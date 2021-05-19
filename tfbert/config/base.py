# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: tokenization_base.py
@date: 2020/09/08
"""
import os
from typing import Dict
import tensorflow.compat.v1 as tf
import copy
import json


class BaseConfig(object):
    filename = "config.json"

    def __init__(self, **kwargs):

        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.use_one_hot_embeddings = kwargs.pop('use_one_hot_embeddings', False)

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                tf.logging.info("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "{} : \n{}".format(self.__class__.__name__, self.to_json_string())

    def to_dict(self):
        """
        将config属性序列化成dict
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """
        将config属性序列化成json字符串
        """
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)

    def save_to_json_file(self, json_file_path):
        """
        将config存入json文件
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        '''
        从json文件读取字典
        :param json_file:
        :return:
        '''
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def from_json_file(cls, json_file: str) -> "BaseConfig":
        """
        从json文件中加载config
        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "BaseConfig":
        """
        从字典中加载config
        """
        config = cls(**config_dict)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        return config

    def save_pretrained(self, save_dir_or_file):
        '''
        保存config，如果save_dir_or_file是个文件夹，
        则保存至默认文件名：save_dir_or_file + 'config.json'
        如果是文件名，保存至该文件
        :param save_dir_or_file:
        :return:
        '''
        if os.path.isdir(save_dir_or_file):
            output_config_file = os.path.join(save_dir_or_file, self.filename)
        else:
            output_config_file = save_dir_or_file

        self.save_to_json_file(output_config_file)
        tf.logging.info('  Configuration saved in {}'.format(output_config_file))
        return output_config_file

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
        elif os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            raise ValueError('Config path should be a directory or file')

        config_dict = cls._dict_from_json_file(config_file)
        return cls.from_dict(config_dict, **kwargs)
