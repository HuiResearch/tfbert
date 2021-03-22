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
                 input_mask=None,
                 token_type_ids=None,
                 label_ids=None,
                 dropout_prob=0.1,
                 compute_type=tf.float32
                 ):
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))

        model = MODELS[model_type](
            config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
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
                 input_mask=None,
                 token_type_ids=None,
                 label_ids=None,
                 dropout_prob=0.1,
                 add_crf=False,
                 compute_type=tf.float32):
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))

        model = MODELS[model_type](
            config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
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
                 input_mask=None,
                 token_type_ids=None,
                 label_ids=None,
                 dropout_prob=0.1,
                 compute_type=tf.float32
                 ):
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))

        model = MODELS[model_type](
            config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
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


class MaskedLM:
    def __init__(
            self,
            model_type,
            config,
            is_training,
            input_ids,
            input_mask=None,
            token_type_ids=None,
            masked_lm_ids=None,
            masked_lm_weights=None,
            masked_lm_positions=None,
            compute_type=tf.float32
    ):
        model_type = model_type.lower()
        if model_type not in MODELS:
            raise ValueError("Unsupported model option: {}, "
                             "you can choose one of {}".format(model_type, "、".join(MODELS.keys())))

        model = MODELS[model_type](
            config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
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
            input_mask=None,
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

        model = MODELS[model_type](
            config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
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
