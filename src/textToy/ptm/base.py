# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: base.py
@date: 2020/09/08
"""
import copy
import tensorflow.compat.v1 as tf


class BaseModel(object):
    def __init__(self, config, is_training):

        self.config = copy.deepcopy(config)

        if isinstance(is_training, bool):
            if not is_training:
                self.config.hidden_dropout_prob = 0.0
                self.config.attention_probs_dropout_prob = 0.0
        else:
            self.config.hidden_dropout_prob = tf.cond(is_training,
                                                      lambda: float(config.hidden_dropout_prob),
                                                      lambda: 0.0)
            self.config.attention_probs_dropout_prob = tf.cond(is_training,
                                                               lambda: float(config.attention_probs_dropout_prob),
                                                               lambda: 0.0)

        self.outputs = ()
        self.embedding_output = None
        self.embedding_table = None

    def get_pooled_output(self):
        return self.outputs[0]

    def get_sequence_output(self):
        return self.outputs[1]

    def get_outputs(self):
        return self.outputs

    def get_all_encoder_layers(self):
        if self.config.output_hidden_states:
            return self.outputs[2]
        else:
            raise ValueError('Please set {} with value {} for config'.format('output_hidden_states', 'True'))

    def get_all_attention_probs(self):
        if self.config.output_attentions:
            return self.outputs[3]
        else:
            raise ValueError('Please set {} with value {} for config'.format('output_attentions', 'True'))

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table
