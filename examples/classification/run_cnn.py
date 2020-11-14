# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: run_cnn.py
@date: 2020/09/21
"""
import os
import platform
import tensorflow.compat.v1 as tf
from textToy.nn import Conv2D, MaxPool
from textToy.ptm.embedding import create_embedding
from textToy.data import process_dataset
from textToy.loss import cross_entropy_loss
from textToy.ptm.utils import get_dropout_prob, create_initializer
from textToy.optimizer import create_optimizer
import pandas as pd
from collections import Counter
from textToy import Trainer, set_seed, ProgressBar, get_devices
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import random
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

set_seed(42)

if platform.system() == 'Windows':
    bar_fn = ProgressBar
else:
    bar_fn = tqdm


class TextCNN:
    def __init__(self,
                 is_training,
                 input_ids, label_ids, seq_length,
                 filter_sizes, num_filters, vocab_size,
                 embedding_dim, num_classes, dropout=0.3):
        embedding_table = create_embedding(
            [vocab_size, embedding_dim], "word_embedding", 0.02
        )
        embedding = tf.nn.embedding_lookup(embedding_table, input_ids)
        embedding = tf.expand_dims(embedding, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv_{}".format(filter_size)):
                filter_shape = [filter_size, embedding_dim, 1, num_filters]
                h = Conv2D(embedding, filter_shape).output
                pooled = MaxPool(h, ksize=[1, seq_length - filter_size + 1, 1, 1]).output
                pooled_outputs.append(pooled)
        conv_output = tf.concat(pooled_outputs, 3)
        conv_output = tf.reshape(conv_output, [-1, num_filters * len(filter_sizes)])

        with tf.variable_scope("classifier"):
            dropout = get_dropout_prob(is_training, dropout_prob=dropout)
            conv_output = tf.nn.dropout(conv_output, rate=dropout)
            self.logits = tf.layers.dense(
                conv_output,
                num_classes,
                kernel_initializer=create_initializer(0.02)
            )
            if label_ids is not None:
                self.loss = cross_entropy_loss(self.logits, label_ids, num_classes)


def create_dataset(filename, vocab2id, label2id, max_len, batch_size, set_type='train'):
    datas = pd.read_csv(filename, encoding='utf-8', sep='\t').values.tolist()
    if set_type == 'train':
        random.shuffle(datas)
    ids = []
    labels = []
    for data in datas:
        label, text = data
        id_ = list(map(lambda x: vocab2id[x] if x in vocab2id else vocab2id['<UNK>'], list(text)))
        id_ = id_[:max_len]
        id_ += [vocab2id["<PAD>"]] * (max_len - len(id_))
        ids.append(id_)
        labels.append(label2id[label])

    def gen():
        for input_id, label_id in zip(ids, labels):
            yield {
                "input_ids": input_id,
                'label_ids': label_id,
            }

    output_types = {"input_ids": tf.int32,
                    'label_ids': tf.int64}
    output_shapes = {"input_ids": tf.TensorShape([None]),
                     'label_ids': tf.TensorShape([])}
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types,
        output_shapes
    )
    return process_dataset(dataset, batch_size, len(ids), set_type)


def create_vocab(filename, max_vocab_size=5000):
    datas = pd.read_csv(filename, encoding='utf-8', sep='\t').values.tolist()
    words = []
    for data in datas:
        words.extend(list(data[1]))
    words = [word.strip() for word in words if word.strip()]
    counter = Counter(words)
    words = counter.most_common(max_vocab_size)
    vocabs = ["<PAD>", "<UNK>"] + [word[0] for word in words]
    vocab2id = dict(zip(vocabs, range(len(vocabs))))
    return vocab2id


def predict(trainer, steps, set_type='dev'):
    # 预测时，先初始化 dataset，参数传入'dev' or 'test'
    trainer.init_iterator(set_type)
    predictions = None
    output_label_ids = None
    for _ in tqdm(range(steps)):
        if set_type == 'dev':
            # 验证会返回loss
            loss, pred, label_id = trainer.eval_step()
        else:
            # 预测不会返回loss
            pred, label_id = trainer.test_step()
        if predictions is None:
            predictions = pred
            output_label_ids = label_id
        else:
            predictions = np.append(predictions, pred, 0)
            output_label_ids = np.append(output_label_ids, label_id, 0)
    predictions = np.argmax(predictions, -1)
    return output_label_ids, predictions


def get_model_fn(max_seq_len):
    def model_fn(inputs, is_training):
        model = TextCNN(
            is_training=is_training,
            input_ids=inputs['input_ids'],
            label_ids=inputs['label_ids'],
            seq_length=max_seq_len,
            filter_sizes=[1, 3, 5, 7], num_filters=128,
            vocab_size=len(vocab2id),
            embedding_dim=128, num_classes=len(labels), dropout=0.3)
        return {'loss': model.loss / gradient_accumulation_steps,
                'outputs': [model.logits, inputs['label_ids']]}

    return model_fn


if __name__ == '__main__':
    max_seq_length = 32
    batch_size = 32
    epochs = 10
    logging_steps = 1000
    output_dir = "ckpt/cnn"
    gradient_accumulation_steps = 1  # 梯度累积步数
    learning_rate = 1e-4
    use_xla = True  # 开启xla加速
    use_torch_mode = False  # torch模式训练会先backward，然后再train，这样可以支持梯度累积，但是速度会慢
    if not use_torch_mode and gradient_accumulation_steps > 1:
        raise ValueError("if you want to use gradient accumulation, please consume use_torch_mode is True.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size = batch_size * len(get_devices())

    labels = ['体育', '娱乐', '家居', '房产', '教育']
    label2id = dict(zip(labels, range(len(labels))))
    vocab2id = create_vocab("data/classification/train.csv")

    train_dataset, num_train_batch = create_dataset(
        "data/classification/train.csv", vocab2id, label2id, max_seq_length, batch_size, set_type='train'
    )
    dev_dataset, num_dev_batch = create_dataset(
        "data/classification/dev.csv", vocab2id, label2id, max_seq_length, batch_size, set_type='dev'
    )
    test_dataset, num_test_batch = create_dataset(
        "data/classification/test.csv", vocab2id, label2id, max_seq_length, batch_size, set_type='test'
    )
    output_types = {"input_ids": tf.int32,
                    'label_ids': tf.int64}
    output_shapes = {"input_ids": tf.TensorShape([None, None]),
                     'label_ids': tf.TensorShape([None])}

    trainer = Trainer(
        'cnn', output_types, output_shapes, device='gpu', use_xla=use_xla, use_torch_mode=use_torch_mode)

    trainer.build_model(get_model_fn(max_seq_length))

    t_total = num_train_batch * epochs // gradient_accumulation_steps

    train_op = create_optimizer(
        init_lr=learning_rate,
        gradients=trainer.gradients,
        variables=trainer.variables,
        num_train_steps=t_total,
        num_warmup_steps=t_total * 0.1
    )

    trainer.compile(train_op, max_checkpoints=1)
    trainer.build_handle(train_dataset, 'train')
    trainer.build_handle(dev_dataset, 'dev')
    trainer.build_handle(test_dataset, 'test')

    trainer.init_variables()

    best_score = 0.

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num epochs = {}".format(epochs))
    tf.logging.info("  batch size = {}".format(batch_size))
    tf.logging.info("  Gradient Accumulation steps = {}".format(gradient_accumulation_steps))
    tf.logging.info("  Total train batch size (accumulation) = {}".format(batch_size * gradient_accumulation_steps))
    tf.logging.info("  optimizer steps = %d", t_total)
    tf.logging.info("  Num devices = {}".format(trainer.num_devices))
    tf.logging.info("  Num params = {}".format(trainer.num_params))

    for epoch in range(epochs):
        epoch_iter = bar_fn(range(num_train_batch), desc='epoch {} '.format(epoch + 1))
        for step in epoch_iter:
            if use_torch_mode:
                train_loss = trainer.backward()
            else:
                train_loss = trainer.train_step()
            epoch_iter.set_description(desc='epoch {} ,loss {:.4f}'.format(epoch + 1, train_loss))
            epoch_iter.set_description(desc='epoch {} ,loss {:.4f}'.format(epoch + 1, train_loss))

            if (step + 1) % gradient_accumulation_steps == 0:
                if use_torch_mode:
                    trainer.train_step()
                    trainer.zero_grad()
                if trainer.global_step % logging_steps == 0 or trainer.global_step == t_total:
                    y_true, y_pred = predict(trainer, num_dev_batch, 'dev')
                    acc = accuracy_score(y_true, y_pred)
                    if acc > best_score:
                        best_score = acc
                        trainer.save_pretrained(output_dir)
                    tf.logging.info("***** eval results *****")
                    tf.logging.info(" global step : {}".format(trainer.global_step))
                    tf.logging.info(" eval accuracy : {:.4f}".format(acc))
                    tf.logging.info(" best accuracy : {:.4f}".format(best_score))

    tf.logging.info("***** Running Test *****")
    trainer.from_pretrained(output_dir)
    y_true, y_pred = predict(trainer, num_test_batch, 'test')
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    tf.logging.info("***** test results *****")
    report = report.split('\n')
    for r in report:
        tf.logging.info(r)
