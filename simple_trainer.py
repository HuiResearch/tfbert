# -*- coding:utf-8 -*-
# @FileName  :run_classifier.py
# @Time      :2021/4/14 19:45
# @Author    :huanghui
import numpy as np

from tfbert.data import SimpleDataset
from tfbert.models import create_word_embeddings, dropout, create_initializer
from tfbert.models.loss import cross_entropy_loss
from tfbert.models.layers import conv2d_layer, max_pooling_layer
import tensorflow.compat.v1 as tf
from tfbert import SimplerTrainer, ProgressBar, set_seed
import pandas as pd
from collections import Counter
from tqdm import tqdm, trange
import platform
from sklearn.metrics import accuracy_score

if platform.system() == 'Windows':
    bar_fn = ProgressBar
else:
    bar_fn = tqdm

set_seed(42)


class TextCNN:
    def __init__(self,
                 max_seq_length,
                 vocab_size,
                 is_training,
                 input_ids,
                 label_ids):
        embedding, _ = create_word_embeddings(
            input_ids=input_ids, vocab_size=vocab_size, embedding_size=300
        )
        embedding = tf.expand_dims(embedding, -1)
        pooled_outputs = []
        for i, filter_size in enumerate([2, 3, 5]):
            with tf.variable_scope("conv_{}".format(filter_size)):
                filter_shape = [filter_size, 300, 1, 128]
                h = conv2d_layer(embedding, filter_shape)
                pooled = max_pooling_layer(h, ksize=[1, max_seq_length - filter_size + 1, 1, 1])
                pooled_outputs.append(pooled)
        conv_output = tf.concat(pooled_outputs, 3)
        conv_output = tf.reshape(conv_output, [-1, 128 * 3])

        with tf.variable_scope("classifier"):
            # dropout = get_dropout_prob(is_training, dropout_prob=dropout)
            if is_training:
                conv_output = dropout(conv_output, dropout_prob=0.3)
            self.logits = tf.layers.dense(
                conv_output,
                5,
                kernel_initializer=create_initializer(0.02),
                name='logits'
            )
            if label_ids is not None:
                self.loss = cross_entropy_loss(self.logits, label_ids, 5)


def get_model_fn(is_training, vocab_size):
    def model_fn():
        input_ids = tf.placeholder(shape=[None, 32], dtype=tf.int64, name='input_ids')
        if is_training:
            label_ids = tf.placeholder(shape=[None], dtype=tf.int64, name='label_ids')
        else:
            label_ids = None
        model = TextCNN(
            32, vocab_size,
            is_training=is_training,
            input_ids=input_ids,
            label_ids=label_ids)
        inputs = {'input_ids': input_ids}

        outputs = {"logits": model.logits}
        if is_training:
            outputs['loss'] = model.loss
            inputs['label_ids'] = label_ids
            outputs['label_ids'] = label_ids
        return inputs, outputs

    return model_fn


def create_vocab(train_file, dev_file):
    datas = pd.read_csv(train_file, encoding='utf-8', sep='\t').values.tolist()
    datas.extend(
        pd.read_csv(dev_file, encoding='utf-8', sep='\t').values.tolist()
    )
    words = []
    for data in datas:
        words.extend(list(data[1]))
    words = [word.strip() for word in words if word.strip()]
    counter = Counter(words)
    words = counter.most_common(5000)
    vocabs = ["<PAD>", "<UNK>"] + [word[0] for word in words]
    vocab2id = dict(zip(vocabs, range(len(vocabs))))
    return vocab2id


def load_data(filename, vocab2id, label2id):
    data = pd.read_csv(filename, encoding='utf-8', sep='\t').values.tolist()
    examples = []
    for d in data:
        label, text = d
        id_ = list(map(lambda x: vocab2id[x] if x in vocab2id else vocab2id['<UNK>'], list(text)))
        id_ = id_[:32]
        id_ += [vocab2id["<PAD>"]] * (32 - len(id_))
        examples.append({'input_ids': id_, 'label_ids': label2id[label]})
    return examples


data_dir = "D:/python/data/data/classification"
vocab2id = create_vocab(data_dir + "/train.csv", data_dir + "/dev.csv")
label2id = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4}

train_data = load_data(data_dir + "/train.csv", vocab2id, label2id)
dev_data = load_data(data_dir + "/dev.csv", vocab2id, label2id)
train_dataset = SimpleDataset(train_data, batch_size=64, is_training=True, padding=True,
                              max_length=32, pad_id=vocab2id['<PAD>'])
dev_dataset = SimpleDataset(dev_data, batch_size=64, is_training=False, padding=True,
                            max_length=32, pad_id=vocab2id['<PAD>'])
trainer = SimplerTrainer(
    optimizer_type='adamw',
    learning_rate=5e-5
)
trainer.build_model(model_fn=get_model_fn(True, len(vocab2id)))
trainer.compile()
trainer.init_variables()
best_score = 0
for epoch in trange(5):
    epoch_iter = bar_fn(train_dataset)
    for d in epoch_iter:
        loss = trainer.train_step(d)
        epoch_iter.set_description(desc='epoch {} ,loss {:.4f}'.format(epoch + 1, loss))
    epoch_iter.close()
    outputs = trainer.predict(dev_dataset.get_all_features(), output_names=['logits', 'label_ids'])
    y_true, y_pred = outputs['label_ids'], np.argmax(outputs['logits'], axis=-1)
    score = accuracy_score(y_true, y_pred)
    if score > best_score:
        best_score = score
        trainer.save_pretrained('output')
    print()
    print(score)
