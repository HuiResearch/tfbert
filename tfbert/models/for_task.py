# -*- coding:utf-8 -*-
# @FileName  :for_task.py
# @Time      :2021/1/31 19:46
# @Author    :huanghui
from . import MODELS
from .model_utils import create_initializer
from . import loss, layers, model_utils
import tensorflow.compat.v1 as tf


class SequenceClassification:
    def __init__(self,
                 model_type,
                 config,
                 num_classes,
                 is_training,
                 input_ids,
                 pinyin_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 label_ids=None,
                 dropout_prob=0.1,
                 compute_type=tf.float32
                 ):
        """
        文本分类基本模型
        :param model_type: 预训练模型类型
        :param config: 配置config
        :param num_classes: 分类类别数
        :param is_training: bool，是否训练，影响dropout设置
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param label_ids:
        :param dropout_prob:
        :param compute_type:
        """
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids}
        if model_type == 'glyce_bert':
            kwargs['pinyin_ids'] = pinyin_ids

        model = MODELS[model_type](
            config,
            is_training=is_training,
            **kwargs,
            compute_type=compute_type
        )
        pooled_output = model.get_pooled_output()
        with tf.variable_scope("classification"):
            # 根据is_training 判断dropout
            if is_training:
                pooled_output = model_utils.dropout(
                    pooled_output,
                    dropout_prob=dropout_prob)
            self.logits = tf.layers.dense(
                pooled_output,
                num_classes,
                kernel_initializer=create_initializer(config.initializer_range))
            if label_ids is not None:
                self.loss = loss.cross_entropy_loss(self.logits, label_ids, num_classes)


class TokenClassification:
    def __init__(self,
                 model_type,
                 config,
                 num_classes,
                 is_training,
                 input_ids,
                 pinyin_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 label_ids=None,
                 dropout_prob=0.1,
                 add_crf=False,
                 compute_type=tf.float32):
        """
        命名实体识别系列模型
        :param model_type: 预训练模型类型
        :param config: 配置config
        :param num_classes: token分类类别数
        :param is_training: bool，是否训练，影响dropout设置
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param label_ids:
        :param dropout_prob:
        :param add_crf: 是否增加crf层
        :param compute_type:
        """
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids}
        if model_type == 'glyce_bert':
            kwargs['pinyin_ids'] = pinyin_ids

        model = MODELS[model_type](
            config,
            is_training=is_training,
            **kwargs,
            compute_type=compute_type
        )
        sequence_output = model.get_sequence_output()
        with tf.variable_scope("classification"):
            if is_training:
                sequence_output = model_utils.dropout(sequence_output,
                                                      dropout_prob=dropout_prob)

            _, seq_len, hidden_size = model_utils.get_shape_list(sequence_output, expected_rank=3)
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
                crf_output = layers.crf_layer(
                    self.logits, num_labels=num_classes,
                    labels=label_ids, lengths=lengths)
                if label_ids is None:
                    self.predictions = crf_output[0]
                else:
                    self.loss, self.predictions = crf_output[0], crf_output[1]
            else:
                probabilities = tf.nn.softmax(self.logits, axis=-1)
                self.predictions = tf.argmax(probabilities, axis=-1)
                if label_ids is not None:
                    self.loss = loss.cross_entropy_loss(self.logits, label_ids, num_classes)


class MultiLabelClassification:
    def __init__(self,
                 model_type,
                 config,
                 num_classes,
                 is_training,
                 input_ids,
                 pinyin_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 label_ids=None,
                 dropout_prob=0.1,
                 compute_type=tf.float32
                 ):
        """
        多标签分类基本模型
        :param model_type: 预训练模型类型
        :param config: 配置config
        :param num_classes: 分类类别数
        :param is_training: bool，是否训练，影响dropout设置
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param label_ids:
        :param dropout_prob:
        :param compute_type:
        """
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids}
        if model_type == 'glyce_bert':
            kwargs['pinyin_ids'] = pinyin_ids

        model = MODELS[model_type](
            config,
            is_training=is_training,
            **kwargs,
            compute_type=compute_type
        )
        pooled_output = model.get_pooled_output()
        with tf.variable_scope("classification"):
            if is_training:
                pooled_output = model_utils.dropout(
                    pooled_output,
                    dropout_prob=dropout_prob)
            logits = tf.layers.dense(
                pooled_output,
                num_classes,
                kernel_initializer=create_initializer(config.initializer_range))
            self.predictions = tf.nn.sigmoid(logits)
            if label_ids is not None:
                per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                           labels=tf.cast(label_ids, tf.float32))
                self.loss = tf.reduce_mean(per_example_loss)


class QuestionAnswering:
    def __init__(self,
                 model_type,
                 config,
                 is_training,
                 input_ids,
                 pinyin_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 start_position=None,
                 end_position=None,
                 dropout_prob=0.1,
                 compute_type=tf.float32):
        """
        阅读理解基本模型
        :param model_type: 预训练模型类型
        :param config: 配置config
        :param is_training: bool，是否训练，影响dropout设置
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param start_position:
        :param end_position:
        :param dropout_prob:
        :param compute_type:
        """
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids}
        if model_type == 'glyce_bert':
            kwargs['pinyin_ids'] = pinyin_ids

        model = MODELS[model_type](
            config,
            is_training=is_training,
            **kwargs,
            compute_type=compute_type
        )
        sequence_output = model.get_sequence_output()
        final_hidden_shape = model_utils.get_shape_list(sequence_output, expected_rank=3)
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]
        with tf.variable_scope("qa_outputs"):
            if is_training:
                sequence_output = model_utils.dropout(sequence_output,
                                                      dropout_prob=dropout_prob)
            output_weights = tf.get_variable(
                "kernel", [2, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "bias", [2], initializer=tf.zeros_initializer())

            final_hidden_matrix = tf.reshape(sequence_output,
                                             [batch_size * seq_length, hidden_size])
            logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            logits = tf.reshape(logits, [batch_size, seq_length, 2])
            logits = tf.transpose(logits, [2, 0, 1])

            unstacked_logits = tf.unstack(logits, axis=0)

            self.start_logits, self.end_logits = unstacked_logits[:2]
            if start_position is not None and end_position is not None:
                seq_length = model_utils.get_shape_list(input_ids)[1]
                start_loss = loss.cross_entropy_loss(self.start_logits, start_position, seq_length)
                end_loss = loss.cross_entropy_loss(self.end_logits, end_position, seq_length)
                self.loss = (start_loss + end_loss) / 2.0


class MaskedLM:
    def __init__(
            self,
            model_type,
            config,
            is_training,
            input_ids,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            masked_lm_ids=None,
            masked_lm_weights=None,
            masked_lm_positions=None,
            compute_type=tf.float32
    ):
        """
        mask 任务模型
        :param model_type: 预训练模型类型
        :param config: 配置config
        :param is_training: bool，是否训练，影响dropout设置
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param masked_lm_ids:
        :param masked_lm_weights:
        :param masked_lm_positions:
        :param compute_type:
        """
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids}
        if model_type == 'glyce_bert':
            kwargs['pinyin_ids'] = pinyin_ids

        model = MODELS[model_type](
            config,
            is_training=is_training,
            **kwargs,
            compute_type=compute_type
        )
        sequence_output = model.get_sequence_output()
        embedding_table = model.get_embedding_table()

        self.prediction_scores = layers.mlm_layer(
            config, sequence_output, embedding_table, masked_lm_positions, scope="cls/predictions"
        )

        if all([el is not None for el in [masked_lm_ids, masked_lm_weights, masked_lm_positions]]):
            self.loss = loss.mlm_loss(self.prediction_scores, masked_lm_ids, config.vocab_size, masked_lm_weights)


class PretrainingLM:
    def __init__(
            self,
            model_type,
            config,
            is_training,
            input_ids,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            masked_lm_ids=None,
            masked_lm_weights=None,
            masked_lm_positions=None,
            next_sentence_labels=None,
            compute_type=tf.float32
    ):
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids}
        if model_type == 'glyce_bert':
            kwargs['pinyin_ids'] = pinyin_ids

        model = MODELS[model_type](
            config,
            is_training=is_training,
            **kwargs,
            compute_type=compute_type
        )
        sequence_output = model.get_sequence_output()
        pooled_output = model.get_pooled_output()
        embedding_table = model.get_embedding_table()
        self.prediction_scores = layers.mlm_layer(
            config, sequence_output, embedding_table, masked_lm_positions, scope="cls/predictions"
        )
        self.seq_relationship_score = layers.seq_rel_weight(config, pooled_output, scope='cls/seq_relationship')

        if all([el is not None for el in
                [masked_lm_ids, masked_lm_weights, masked_lm_positions]]) and next_sentence_labels is not None:
            masked_lm_loss = loss.mlm_loss(self.prediction_scores, masked_lm_ids, config.vocab_size, masked_lm_weights)
            next_sentence_loss = loss.cross_entropy_loss(self.seq_relationship_score, next_sentence_labels, 2)
            self.loss = masked_lm_loss + next_sentence_loss
