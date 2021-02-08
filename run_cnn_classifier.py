# -*- coding:utf-8 -*-
# @FileName  :run_cnn_classifier.py
# @Time      :2021/2/1 20:40
# @Author    :huanghui
"""
没有使用混合精度的写法，用简单的textCNN为例
"""
import os
import platform
import tensorflow.compat.v1 as tf
import argparse
from tfbert.models.layers import conv2d_layer, max_pooling_layer
from tfbert.models.embeddings import create_word_embeddings
from tfbert.data import process_dataset
from tfbert.models.loss import cross_entropy_loss
from tfbert.models.model_utils import create_initializer, dropout
from tfbert import Trainer, set_seed, ProgressBar, devices, create_optimizer
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import random
import numpy as np

set_seed(42)

if platform.system() == 'Windows':
    bar_fn = ProgressBar
else:
    bar_fn = tqdm


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer_type', default='adamw', type=str, help="优化器类型")
    parser.add_argument('--output_dir', default='output/checkpoint', type=str, help="")
    parser.add_argument('--export_dir', default='output/pb', type=str, help="")

    parser.add_argument('--labels', default='体育,娱乐,家居,房产,教育', type=str, help="文本分类标签")
    parser.add_argument('--train_file', default='data/classification/train.csv', type=str, help="")
    parser.add_argument('--dev_file', default='data/classification/dev.csv', type=str, help="")
    parser.add_argument('--test_file', default='data/classification/test.csv', type=str, help="")

    parser.add_argument("--num_train_epochs", default=3, type=int, help="训练轮次")
    parser.add_argument("--max_vocab_size", default=5000, type=int, help="词表最大数量")
    parser.add_argument("--max_seq_length", default=32, type=int, help="最大句子长度")
    parser.add_argument("--embedding_dim", default=128, type=int, help="词嵌入维度")
    parser.add_argument("--num_filters", default=128, type=int, help="卷积核数量")
    parser.add_argument('--filter_sizes', default='2,3,4', type=str, help="卷积核尺寸")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
    parser.add_argument("--batch_size", default=32, type=int, help="训练批次")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累积")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="学习率")
    parser.add_argument("--warmup_proportion", default=0.1, type=int,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--weight_decay", default=0.01, type=int, help="Weight decay if we apply some.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set.")
    parser.add_argument("--do_export", action="store_true", help="将模型导出为pb格式.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="是否边训练边验证")

    parser.add_argument("--logging_steps", default=1000, type=int, help="训练时每隔几步验证一次")
    parser.add_argument("--save_steps", default=1000, type=int, help="训练时每隔几步保存一次")
    parser.add_argument("--random_seed", default=42, type=int, help="随机种子")
    parser.add_argument("--threads", default=8, type=int, help="数据处理进程数")
    parser.add_argument("--max_checkpoints", default=1, type=int, help="模型保存最大数量，默认只保存一个")
    parser.add_argument("--single_device", action="store_true", help="是否只使用一个device，默认使用所有的device训练")
    parser.add_argument("--use_xla", action="store_true", help="是否使用XLA加速")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not args.single_device:
        args.batch_size = args.batch_size * len(devices())

    args.labels = args.labels.split(',')
    args.filter_sizes = list(map(lambda x: int(x), args.filter_sizes.split(',')))
    return args


class TextCNN:
    def __init__(self,
                 args,
                 is_training,
                 input_ids,
                 label_ids):
        embedding, _ = create_word_embeddings(
            input_ids=input_ids, vocab_size=args.vocab_size, embedding_size=args.embedding_dim
        )
        embedding = tf.expand_dims(embedding, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(args.filter_sizes):
            with tf.variable_scope("conv_{}".format(filter_size)):
                filter_shape = [filter_size, args.embedding_dim, 1, args.num_filters]
                h = conv2d_layer(embedding, filter_shape)
                pooled = max_pooling_layer(h, ksize=[1, args.max_seq_length - filter_size + 1, 1, 1])
                pooled_outputs.append(pooled)
        conv_output = tf.concat(pooled_outputs, 3)
        conv_output = tf.reshape(conv_output, [-1, args.num_filters * len(args.filter_sizes)])

        with tf.variable_scope("classifier"):
            # dropout = get_dropout_prob(is_training, dropout_prob=dropout)
            if is_training:
                conv_output = dropout(conv_output, dropout_prob=args.dropout)
            self.logits = tf.layers.dense(
                conv_output,
                len(args.labels),
                kernel_initializer=create_initializer(0.02),
                name='logits'
            )
            if label_ids is not None:
                self.loss = cross_entropy_loss(self.logits, label_ids, len(args.labels))


def return_types_and_shapes(for_trainer=False):
    if for_trainer:
        shape = tf.TensorShape([None, None])
        label_shape = tf.TensorShape([None])
    else:
        shape = tf.TensorShape([None])
        label_shape = tf.TensorShape([])

    input_types = {"input_ids": tf.int32,
                   'label_ids': tf.int64}
    input_shapes = {"input_ids": shape,
                    'label_ids': label_shape}
    return input_types, input_shapes


def create_dataset(set_type, args, vocab2id):
    label2id = dict(zip(args.labels, range(len(args.labels))))
    filename_map = {
        'train': args.train_file, 'dev': args.dev_file, 'test': args.test_file
    }
    datas = pd.read_csv(filename_map[set_type], encoding='utf-8', sep='\t').values.tolist()
    if set_type == 'train':
        random.shuffle(datas)
    ids = []
    labels = []
    for data in datas:
        label, text = data
        id_ = list(map(lambda x: vocab2id[x] if x in vocab2id else vocab2id['<UNK>'], list(text)))
        id_ = id_[:args.max_seq_length]
        id_ += [vocab2id["<PAD>"]] * (args.max_seq_length - len(id_))
        ids.append(id_)
        labels.append(label2id[label])

    def gen():
        for input_id, label_id in zip(ids, labels):
            yield {
                "input_ids": input_id,
                'label_ids': label_id,
            }

    input_types, input_shapes = return_types_and_shapes()
    dataset = tf.data.Dataset.from_generator(
        gen,
        input_types,
        input_shapes
    )
    return process_dataset(dataset, args.batch_size, len(ids), set_type)


def create_vocab(args):
    datas = pd.read_csv(args.train_file, encoding='utf-8', sep='\t').values.tolist()
    datas.extend(
        pd.read_csv(args.dev_file, encoding='utf-8', sep='\t').values.tolist()
    )
    words = []
    for data in datas:
        words.extend(list(data[1]))
    words = [word.strip() for word in words if word.strip()]
    counter = Counter(words)
    words = counter.most_common(args.max_vocab_size)
    vocabs = ["<PAD>", "<UNK>"] + [word[0] for word in words]
    open(os.path.join(args.output_dir, 'vocab.txt'), 'w', encoding='utf-8').write(
        "\n".join(vocabs)
    )
    vocab2id = dict(zip(vocabs, range(len(vocabs))))
    return vocab2id


def load_vocab(vocab_path):
    vocabs = open(vocab_path, 'r', encoding='utf-8').read().split('\n')
    vocab2id = dict(zip(vocabs, range(len(vocabs))))
    return vocab2id


def predict(trainer: Trainer, steps, set_type='dev'):
    # 预测时，先初始化 dataset，参数传入'dev' or 'test'
    outputs = trainer.predict(set_type, ['logits', 'label_ids'], steps)
    predictions = np.argmax(outputs['logits'], -1)
    return outputs['label_ids'], predictions


def get_model_fn(args):
    def model_fn(inputs, is_training):
        model = TextCNN(
            args,
            is_training=is_training,
            input_ids=inputs['input_ids'],
            label_ids=inputs['label_ids'])
        return {'loss': model.loss / args.gradient_accumulation_steps,
                'outputs': {'logits': model.logits, 'label_ids': inputs['label_ids']}}

    return model_fn


def get_serving_fn(args):
    """
    定义serving的fn，fn需要可以不传参调用，返回的是pb模型的inputs和outputs
    inputs包含模型接收的输入，outputs包含模型的输出，都是字典形式
    :param args:
    :return:
    """

    def serving_fn():
        input_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='input_ids')
        model = TextCNN(
            args,
            is_training=False,
            input_ids=input_ids,
            label_ids=None)
        inputs = {'input_ids': input_ids}
        outputs = {"logits": model.logits}
        return inputs, outputs

    return serving_fn


def train(trainer: Trainer, args, vocab2id):
    train_dataset, train_batch_one_epoch = create_dataset('train', args, vocab2id)
    if args.evaluate_during_training:
        dev_dataset, dev_batch_one_epoch = create_dataset('dev', args, vocab2id)
    else:
        dev_dataset, dev_batch_one_epoch = None, None

    t_total = train_batch_one_epoch * args.num_train_epochs // args.gradient_accumulation_steps

    optimizer = create_optimizer(
        args.learning_rate,
        num_train_steps=t_total,
        num_warmup_steps=t_total * args.warmup_proportion,
        optimizer_type=args.optimizer_type
    )
    trainer.compile(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad=1.0,
        max_checkpoints=1,
        optimizer=optimizer
    )
    trainer.init_variables()
    trainer.prepare_dataset(train_dataset, 'train')
    if dev_dataset is not None:
        trainer.prepare_dataset(dev_dataset, 'dev')

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num epochs = {}".format(args.num_train_epochs))
    tf.logging.info("  batch size = {}".format(args.batch_size))
    tf.logging.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    tf.logging.info("  Total train batch size (accumulation) = {}".format(
        args.batch_size * args.gradient_accumulation_steps))
    tf.logging.info("  optimizer steps = %d", t_total)
    tf.logging.info("  Num devices = {}".format(trainer.num_devices))
    tf.logging.info("  Num params = {}".format(trainer.num_params))
    best_score = 0.
    for epoch in range(args.num_train_epochs):
        epoch_iter = bar_fn(range(train_batch_one_epoch), desc='epoch {} '.format(epoch + 1))
        for step in epoch_iter:
            train_loss = trainer.train_step()
            epoch_iter.set_description(desc='epoch {} ,loss {:.4f}'.format(epoch + 1, train_loss))
            if args.evaluate_during_training and trainer.global_step_changed and (
                    trainer.global_step % args.logging_steps == 0 or trainer.global_step == t_total):
                y_true, y_pred = predict(trainer, dev_batch_one_epoch, 'dev')
                acc = accuracy_score(y_true, y_pred)
                if acc > best_score:
                    best_score = acc
                    trainer.save_pretrained(args.output_dir)
                tf.logging.info("***** eval results *****")
                tf.logging.info(" global step : {}".format(trainer.global_step))
                tf.logging.info(" eval accuracy : {:.4f}".format(acc))
                tf.logging.info(" best accuracy : {:.4f}".format(best_score))
            if not args.evaluate_during_training and trainer.global_step_changed and (
                    trainer.global_step % args.save_steps == 0 or trainer.global_step == t_total):
                trainer.save_pretrained(args.output_dir)
        epoch_iter.close()

    tf.logging.info("***** Finished training *****")


def main():
    args = create_args()

    input_types, input_shapes = return_types_and_shapes(for_trainer=True)
    trainer = Trainer(
        input_types=input_types,
        input_shapes=input_shapes,
        use_xla=args.use_xla,
        single_device=args.single_device)
    if args.do_train:
        vocab2id = create_vocab(args)
    else:
        vocab2id = load_vocab(os.path.join(args.output_dir, 'vocab.txt'))
    args.vocab_size = len(vocab2id)
    trainer.build_model(get_model_fn(args))
    if args.do_train:
        train(trainer, args, vocab2id)
    if args.do_eval:
        dev_dataset, dev_batch_one_epoch = create_dataset('dev', args, vocab2id)
        trainer.prepare_dataset(dev_dataset, 'dev')
        # 加载保存好的权重
        trainer.from_pretrained(args.output_dir)
        y_true, y_pred = predict(trainer, dev_batch_one_epoch, 'dev')
        report = classification_report(y_true, y_pred, target_names=args.labels, digits=4)
        tf.logging.info("***** eval results *****")
        report = report.split('\n')
        for r in report:
            tf.logging.info(r)
    if args.do_test:
        test_dataset, test_batch_one_epoch = create_dataset('test', args, vocab2id)
        trainer.from_pretrained(args.output_dir)
        trainer.prepare_dataset(test_dataset, 'test')
        outputs = trainer.predict('test', ['logits'], test_batch_one_epoch)
        label_ids = np.argmax(outputs['logits'], axis=-1)
        labels = list(map(lambda x: args.labels[x], label_ids))
        open(
            os.path.join(args.output_dir, 'prediction.txt'), 'w', encoding='utf-8'
        ).write("\n".join(labels))

    if args.do_export:
        trainer.export(
            get_serving_fn(args),
            args.output_dir,
            args.export_dir
        )


if __name__ == '__main__':
    main()
