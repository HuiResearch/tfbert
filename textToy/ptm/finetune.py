# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: finetune.py
@date: 2020/09/08
"""
import tensorflow.compat.v1 as tf
from .utils import (
    create_initializer, get_dropout_prob, get_shape_list, mlm_weight, seq_rel_weight, gather_indexes)
from .ckpt_utils import init_checkpoints
from ..nn import CRF
from . import MODELS
from ..config import CONFIGS
from ..tokenizer import TOKENIZERS
import numpy as np
from ..loss import mlm_loss, cross_entropy_loss


class TinySequenceClassification:
    def __init__(self,
                 model_type,
                 config,
                 num_classes,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 dropout_prob=0.1
                 ):
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))

        config.output_hidden_states = True
        config.output_attentions = True

        model = MODELS[model_type](
            config, input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            is_training=is_training
        )
        pooled_output = model.get_pooled_output()
        all_encoder_layers = model.get_all_encoder_layers()
        all_attention_probs = model.get_all_attention_probs()
        with tf.variable_scope("classification"):
            # 根据is_training 判断dropout
            dropout_prob = get_dropout_prob(is_training, dropout_prob)

            pooled_output = tf.nn.dropout(pooled_output,
                                          rate=dropout_prob)
            self.logits = tf.layers.dense(
                pooled_output,
                num_classes,
                kernel_initializer=create_initializer(config.initializer_range))


class SequenceClassification:
    def __init__(self,
                 model_type,
                 config,
                 num_classes,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 label_ids=None,
                 dropout_prob=0.1
                 ):
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))

        model = MODELS[model_type](
            config, input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            is_training=is_training
        )
        pooled_output = model.get_pooled_output()
        with tf.variable_scope("classification"):
            # 根据is_training 判断dropout
            dropout_prob = get_dropout_prob(is_training, dropout_prob)

            pooled_output = tf.nn.dropout(pooled_output,
                                          rate=dropout_prob)
            self.logits = tf.layers.dense(
                pooled_output,
                num_classes,
                kernel_initializer=create_initializer(config.initializer_range))
            if label_ids is not None:
                self.loss = cross_entropy_loss(self.logits, label_ids, num_classes)


class TokenClassification:
    def __init__(self,
                 model_type,
                 config,
                 num_classes,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 label_ids=None,
                 dropout_prob=0.1,
                 add_crf=False):
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))

        model = MODELS[model_type](
            config, input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            is_training=is_training
        )
        sequence_output = model.get_sequence_output()
        with tf.variable_scope("classification"):
            # 根据is_training 判断dropout
            dropout_prob = get_dropout_prob(is_training, dropout_prob)

            sequence_output = tf.nn.dropout(sequence_output,
                                            rate=dropout_prob)

            _, seq_len, hidden_size = get_shape_list(sequence_output, expected_rank=3)
            sequence_output = tf.reshape(sequence_output, [-1, hidden_size])
            logits = tf.layers.dense(
                sequence_output,
                num_classes,
                kernel_initializer=create_initializer(config.initializer_range)
            )
            self.logits = tf.reshape(logits, [-1, seq_len, num_classes])
            if add_crf:
                used = tf.sign(tf.abs(input_ids))
                lengths = tf.reduce_sum(used, reduction_indices=1)
                crf = CRF(self.logits, num_labels=num_classes, labels=label_ids, lengths=lengths)
                self.loss = crf.loss
                self.predictions = crf.pred_ids
            else:
                probabilities = tf.nn.softmax(self.logits, axis=-1)
                self.predictions = tf.argmax(probabilities, axis=-1)
                if label_ids is not None:
                    self.loss = cross_entropy_loss(self.logits, label_ids, num_classes)


class MultiLabelClassification:
    def __init__(self,
                 model_type,
                 config,
                 num_classes,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 label_ids=None,
                 dropout_prob=0.1
                 ):
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))

        model = MODELS[model_type](
            config, input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            is_training=is_training
        )
        pooled_output = model.get_pooled_output()
        with tf.variable_scope("classification"):
            # 根据is_training 判断dropout
            dropout_prob = get_dropout_prob(is_training, dropout_prob)

            pooled_output = tf.nn.dropout(pooled_output,
                                          rate=dropout_prob)
            logits = tf.layers.dense(
                pooled_output,
                num_classes,
                kernel_initializer=create_initializer(config.initializer_range))
            self.predictions = tf.nn.sigmoid(logits)
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                       labels=tf.cast(label_ids, tf.float32))
            self.loss = tf.reduce_mean(per_example_loss)


class PretrainingModel:
    def __init__(self,
                 model_type,
                 config,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 mlm_ids=None,
                 mlm_weights=None,
                 mlm_positions=None,
                 nsp_labels=None,
                 is_training=None,
                 mode='mlm'):  # mlm, seq_rel, both

        model = MODELS[model_type](
            config, input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            is_training=is_training
        )

        if mode in ['mlm', 'both']:
            embedding_table = model.get_embedding_table()
            sequence_output = model.get_sequence_output()

            self.mlm_logits = mlm_weight(config, sequence_output, embedding_table, scope='cls/predictions')
            if all([el is not None for el in [mlm_ids, mlm_weights]]):
                mlm_logits = gather_indexes(self.mlm_logits, mlm_positions)
                self.mlm_loss = mlm_loss(mlm_logits, mlm_ids, config.vocab_size, mlm_weights)

        if mode in ['seq_rel', 'both']:
            pooled_output = model.get_pooled_output()
            self.nsp_logits = seq_rel_weight(config, pooled_output, scope='cls/seq_relationship')
            if nsp_labels is not None:
                self.nsp_loss = cross_entropy_loss(self.nsp_logits, nsp_labels, depth=2)


class PTMExtractFeature:
    def __init__(self,
                 model_type,
                 output_type='pool',
                 model_path=None,
                 config_path=None,
                 vocab_path=None
                 ):
        config = CONFIGS[model_type].from_pretrained(config_path)
        self.tokenizer = TOKENIZERS[model_type].from_pretrained(vocab_path, do_lower_case=True)

        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        self.session = tf.Session(config=sess_conf)

        self.iterator = tf.data.Iterator.from_structure(
            self.return_types(), self.return_shape(batch=True))

        inputs = self.iterator.get_next()
        model = MODELS[model_type](
            config, input_ids=inputs['input_ids'],
            input_mask=inputs['input_mask'],
            token_type_ids=inputs['token_type_ids'],
            is_training=False
        )
        init_checkpoints(model_path, model_type, True)
        self.session.run(tf.global_variables_initializer())
        if output_type == 'pool':
            self.output = model.get_pooled_output()
        elif output_type == 'sequence':
            self.output = model.get_sequence_output()
        else:
            raise ValueError('{} is unsupported'.format(output_type))

    @classmethod
    def return_types(cls):
        return {"input_ids": tf.int32,
                "input_mask": tf.int32,
                "token_type_ids": tf.int32}

    @classmethod
    def return_shape(cls, batch=False):
        shape = tf.TensorShape([None, None]) if batch else tf.TensorShape([None])
        return {"input_ids": shape,
                "input_mask": shape,
                "token_type_ids": shape}

    def create_dataset(self, sentences, batch_size=None):
        def gen():
            for sentence in sentences:
                inputs = self.tokenizer.encode(
                    sentence,
                    text_pair=None,
                    add_special_tokens=True,  # 是否增加 cls  sep
                    max_length=512,  # 最大长度
                    pad_to_max_length=False  # 是否将句子padding到最大长度
                )
                yield {
                    "input_ids": [inputs['input_ids']] if batch_size is None else inputs['input_ids'],
                    "input_mask": [inputs['input_mask']] if batch_size is None else inputs['input_mask'],
                    "token_type_ids": [inputs['token_type_ids']] if batch_size is None else inputs['token_type_ids']
                }

        dataset = tf.data.Dataset.from_generator(
            gen,
            self.return_types(),
            self.return_shape(batch=bool(batch_size is None))
        )

        if batch_size is not None:
            # padded_batch 按照batch内最大长度自动padding
            dataset = dataset.padded_batch(
                batch_size, padded_shapes=self.return_shape(False)).prefetch(batch_size)
        else:
            dataset = dataset.prefetch(1)
        return dataset

    def predict(self, sentences, batch_size=None):
        """
        直接输入多个句子进行预测

        example：
        from textToy import PTMExtractFeature

        model = PTMExtractFeature(
            'bert',
            output_type='pool',
            model_path='F:/Chinese-BERT/chinese_rbt3_L-3_H-768_A-12/model.ckpt',
            config_path='F:/Chinese-BERT/chinese_rbt3_L-3_H-768_A-12/config.json',
            vocab_path='F:/Chinese-BERT/chinese_rbt3_L-3_H-768_A-12/vocab.txt'
        )
        while True:
            text = input(">>>")
            print(model.predict([text], batch_size=1))

        :param sentences:
        :param batch_size: 不传入默认一条一条预测，传入的话dataset会批次输出
        :return:
        """
        dataset = self.create_dataset(sentences, batch_size)
        init_op = self.iterator.make_initializer(dataset)
        self.session.run(init_op)
        outputs = None
        while True:
            try:
                pred = self.session.run(self.output)
                if outputs is None:
                    outputs = pred
                else:
                    outputs = np.append(outputs, pred, 0)
            except tf.errors.OutOfRangeError:
                break
        return outputs

    def predict_gen(self, iter_fn):
        """
        传入一个生成器，在线预测，这样可以用来部署
        example：
        extract_fn = PTMExtractFeature(
            'bert',
            output_type='pool',
            model_path='F:/Chinese-BERT/chinese_rbt3_L-3_H-768_A-12/model.ckpt',
            config_path='F:/Chinese-BERT/chinese_rbt3_L-3_H-768_A-12/config.json',
            vocab_path='F:/Chinese-BERT/chinese_rbt3_L-3_H-768_A-12/vocab.txt'
        )
        def input_fn():
            while True:
                text = input()
                yield text

        for res in extract_fn.predict_gen(iter_fn=input_fn()):
            print(res)
        :param iter_fn: 数据生成器
        :return: 一个生成器
        """
        dataset = self.create_dataset(iter_fn)
        init_op = self.iterator.make_initializer(dataset)
        self.session.run(init_op)
        while True:
            try:
                output = self.session.run(self.output)
                yield output
            except tf.errors.OutOfRangeError:
                break
