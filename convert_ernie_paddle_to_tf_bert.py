# -*- coding: UTF-8 -*-
# author : 'huanghui'
#  date  : '2021/5/25 22:05'
# project: 'tfbert'
import json
import os
import paddle
import collections
import numpy as np
import argparse
import tensorflow.compat.v1 as tf
from tfbert import BertConfig


def build_params_map_to_pt(num_layers=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        'word_emb.weight': "bert.embeddings.word_embeddings.weight",
        'pos_emb.weight': "bert.embeddings.position_embeddings.weight",
        'sent_emb.weight': "bert.embeddings.token_type_embeddings.weight",
        'ln.weight': 'bert.embeddings.LayerNorm.gamma',
        'ln.bias': 'bert.embeddings.LayerNorm.beta',
    })
    for i in range(num_layers):
        weight_map[f'encoder_stack.block.{i}.attn.q.weight'] = f'bert.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'encoder_stack.block.{i}.attn.q.bias'] = f'bert.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'encoder_stack.block.{i}.attn.k.weight'] = f'bert.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'encoder_stack.block.{i}.attn.k.bias'] = f'bert.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'encoder_stack.block.{i}.attn.v.weight'] = f'bert.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'encoder_stack.block.{i}.attn.v.bias'] = f'bert.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'encoder_stack.block.{i}.attn.o.weight'] = f'bert.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'encoder_stack.block.{i}.attn.o.bias'] = f'bert.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'encoder_stack.block.{i}.ln1.weight'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'encoder_stack.block.{i}.ln1.bias'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'encoder_stack.block.{i}.ffn.i.weight'] = f'bert.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'encoder_stack.block.{i}.ffn.i.bias'] = f'bert.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'encoder_stack.block.{i}.ffn.o.weight'] = f'bert.encoder.layer.{i}.output.dense.weight'
        weight_map[f'encoder_stack.block.{i}.ffn.o.bias'] = f'bert.encoder.layer.{i}.output.dense.bias'
        weight_map[f'encoder_stack.block.{i}.ln2.weight'] = f'bert.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'encoder_stack.block.{i}.ln2.bias'] = f'bert.encoder.layer.{i}.output.LayerNorm.beta'
    # add pooler
    weight_map.update(
        {
            'pooler.weight': 'bert.pooler.dense.weight',
            'pooler.bias': 'bert.pooler.dense.bias',
            'mlm.weight': 'cls.predictions.transform.dense.weight',
            'mlm.bias': 'cls.predictions.transform.dense.bias',
            'mlm_ln.weight': 'cls.predictions.transform.LayerNorm.gamma',
            'mlm_ln.bias': 'cls.predictions.transform.LayerNorm.beta',
            'mlm_bias': 'cls.predictions.bias'
        }
    )

    return weight_map


def build_config(paddle_config_file):
    ernie_config = json.load(open(paddle_config_file, 'r', encoding='utf-8'))
    if 'sent_type_vocab_size' in ernie_config:
        ernie_config['type_vocab_size'] = ernie_config['sent_type_vocab_size']
    config = BertConfig(
        **ernie_config
    )
    return config


def convert_paddle_checkpoint_to_tf(
        paddle_weight_file, paddle_config_file, paddle_vocab_file, save_dir):
    params = paddle.load(paddle_weight_file)
    config = build_config(paddle_config_file)
    weight_map = build_params_map_to_pt(config.num_hidden_layers)

    var_map = (
        ("layer.", "layer_"),
        ("word_embeddings.weight", "word_embeddings"),
        ("position_embeddings.weight", "position_embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings"),
        (".", "/"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("weight", "kernel"),
    )

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return name

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for pd_name, pd_var in params.items():
            tf_name = to_tf_var_name(weight_map[pd_name])
            pd_tensor = pd_var.numpy()
            tf_var = create_tf_var(tensor=pd_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, pd_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, pd_tensor)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(save_dir, "model.ckpt"))
    config.save_pretrained(save_dir)
    # ernie gram 里边是vocab + \t + id
    with open(paddle_vocab_file, 'r', encoding='utf-8') as f, \
            open(os.path.join(save_dir, "vocab.txt"), 'w', encoding='utf-8') as w:
        for line in f:
            line = line.strip()
            if '\t' in line:
                line = line.split('\t')[0]
            if line:
                w.write(line + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paddle_weight_file", type=str, required=True, help="paddle 权重文件，只支持动态图的权重文件")
    parser.add_argument("--paddle_config_file", type=str, required=True, help="paddle 配置文件名")
    parser.add_argument("--paddle_vocab_file", type=str, required=True, help="paddle 词典文件")
    parser.add_argument(
        "--save_dir", type=str, default=None, required=True, help="转换后权重保存文件夹"
    )
    args = parser.parse_args()
    convert_paddle_checkpoint_to_tf(
        args.paddle_weight_file, args.paddle_config_file, args.paddle_vocab_file,
        args.save_dir
    )


if __name__ == '__main__':
    main()
