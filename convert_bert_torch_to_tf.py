# -*- coding: UTF-8 -*-
# author : 'huanghui'
#  date  : '2021/5/29 8:23'
# project: 'tfbert'
import os
import argparse
import tensorflow.compat.v1 as tf
import numpy as np
import shutil

import torch


def convert_pytorch_checkpoint_to_tf(pt_weight_file, pt_config_file, pt_vocab_file, save_dir: str):
    tensors_to_transpose = (
        "dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")

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

    state_dict = torch.load(pt_weight_file, map_location='cpu')

    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return f"{name}"

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(save_dir, "model.ckpt"))
    if os.path.exists(os.path.join(save_dir, 'checkpoint')):
        try:
            os.remove(os.path.join(save_dir, 'checkpoint'))
            print(
                "We will delete the checkpoint file to avoid errors in loading weights "
                "using tf.train.latest_checkpoint api.")
        except:
            pass
    if pt_config_file is not None and os.path.exists(pt_config_file):
        shutil.copyfile(pt_config_file, os.path.join(save_dir, 'config.json'))
    if pt_vocab_file is not None and os.path.exists(pt_vocab_file):
        shutil.copyfile(pt_vocab_file, os.path.join(save_dir, 'vocab.txt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_model_dir", type=str, default=None, help="pytorch 模型文件所在文件夹")
    parser.add_argument("--pt_weight_file", type=str, default=None, help="pytorch 权重文件")
    parser.add_argument("--pt_config_file", type=str, default=None, help="pytorch 配置文件名")
    parser.add_argument("--pt_vocab_file", type=str, default=None, help="pytorch 词典文件")
    parser.add_argument(
        "--save_dir", type=str, default=None, required=True, help="转换后权重保存文件夹"
    )
    args = parser.parse_args()
    if args.pytorch_model_dir is not None:
        args.pt_weight_file = os.path.join(args.pytorch_model_dir, 'pytorch_model.bin')
        args.pt_config_file = os.path.join(args.pytorch_model_dir, 'config.json')
        args.pt_vocab_file = os.path.join(args.pytorch_model_dir, 'vocab.txt')
    convert_pytorch_checkpoint_to_tf(
        args.pt_weight_file, args.pt_config_file, args.pt_vocab_file,
        args.save_dir
    )


if __name__ == '__main__':
    main()
