# -*- coding:utf-8 -*-
# @FileName  :serving.py
# @Time      :2021/2/3 21:52
# @Author    :huanghui
import os
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.contrib import predictor


def export_model_to_pb(model_name_or_path, export_path,
                       inputs: dict, outputs: dict):
    """
    config = BertConfig.from_pretrained('ckpt')
    input_ids = tf.placeholder(shape=[None, 32], dtype=tf.int64, name='input_ids')
    input_mask = tf.placeholder(shape=[None, 32], dtype=tf.int64, name='input_mask')
    token_type_ids = tf.placeholder(shape=[None, 32], dtype=tf.int64, name='token_type_ids')
    model = model = SequenceClassification(
            model_type='bert',
            config=config,
            num_classes=len(labels),
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids)
     export_model_to_pb('ckpt/model.ckpt-1875', 'pb',
                    inputs={'input_ids': input_ids, 'input_mask': input_mask, 'token_type_ids': token_type_ids},
                    outputs={'logits': model.logits}
                    )
    :param model_name_or_path:
    :param export_path:
    :param inputs:
    :param outputs:
    :return:
    """
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    saver = tf.train.Saver()

    if os.path.isdir(model_name_or_path):
        ckpt_file = tf.train.latest_checkpoint(model_name_or_path)
        if ckpt_file is None:
            ckpt_file = os.path.join(model_name_or_path, 'model.ckpt')
    else:
        ckpt_file = model_name_or_path
    saver.restore(sess, ckpt_file)
    save_pb(sess, export_path, inputs=inputs, outputs=outputs, saver=saver)
    tf.logging.info('export model to {}'.format(export_path))


def save_pb(session, export_dir, inputs, outputs, legacy_init_op=None, saver=None):
    '''
    重写 pb 保存模型接口，可以添加saver，剔除额外参数
    :param session:
    :param export_dir:
    :param inputs:
    :param outputs:
    :param legacy_init_op:
    :param saver:
    :return:
    '''
    signature_def_map = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature_def_utils.predict_signature_def(inputs, outputs)
    }
    b = builder.SavedModelBuilder(export_dir)
    b.add_meta_graph_and_variables(
        session,
        tags=[tag_constants.SERVING],
        signature_def_map=signature_def_map,
        assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
        main_op=legacy_init_op,
        clear_devices=True,
        saver=saver
    )
    b.save()


def load_pb(pb_dir):
    '''
    加载保存的pb模型，该方法适用于线下测试pb模型，部署到线上就需要使用tf serving的方式来预测

    测试例子：
    通用是分类模型。

    predict_fn, input_names, output_names = load_pb('pb')
    tokenizer = BertTokenizer.from_pretrained('ckpt', do_lower_case=True)
    inputs = tokenizer.encode("名人堂故事之威斯康辛先生：大范&联盟总裁的前辈",
                             add_special_tokens=True, max_length=32,
                             pad_to_max_length=True)
    prediction = predict_fn(
    {
        'input_ids': [inputs['input_ids']],
        'input_mask': [inputs['input_mask']],
        'token_type_ids': [inputs['token_type_ids']]
    }
    )
    print(prediction)

    输出{'logits': array([[ 5.1162577, -3.842629 , -0.2090739,  1.629769 , -2.6358554]],
        dtype=float32)}

    :param pb_dir: 保存的pb模型的文件夹
    :return: 预测fn, fn接收的输入的names，fn输出的names
    '''
    predict_fn = predictor.from_saved_model(pb_dir)
    input_names = list(predict_fn._feed_tensors.keys())
    output_names = list(predict_fn._fetch_tensors.keys())
    return predict_fn, input_names, output_names
