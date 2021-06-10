# -*- coding: UTF-8 -*-
__author__ = 'huanghui'
__date__ = '2021/4/18 15:06'
__project__ = 'tfbert'

import json
import os
import argparse
import tensorflow.compat.v1 as tf
from tfbert import (
    Trainer, Dataset,
    SequenceClassification,
    CONFIGS, TOKENIZERS, devices, set_seed)
from tfbert.data.classification import convert_examples_to_features, InputExample
from sklearn.metrics import accuracy_score
import pandas as pd
from typing import Dict
import numpy as np


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='bert', type=str, choices=CONFIGS.keys())
    parser.add_argument('--optimizer_type', default='adamw', type=str, help="优化器类型")
    parser.add_argument('--model_dir', default='model_path', type=str,
                        help="预训练模型存放文件夹，文件夹下ckpt文件名为model.ckpt，"
                             "config文件名为config.json，词典文件名为vocab.txt")

    parser.add_argument('--config_path', default=None, type=str, help="若配置文件名不是默认的，可在这里输入")
    parser.add_argument('--vocab_path', default=None, type=str, help="若词典文件名不是默认的，可在这里输入")
    parser.add_argument('--pretrained_checkpoint_path', default=None, type=str, help="若模型文件名不是默认的，可在这里输入")
    parser.add_argument('--output_dir', default='output/classification', type=str, help="")
    parser.add_argument('--export_dir', default='output/classification/pb', type=str, help="")

    parser.add_argument('--labels', default='体育,娱乐,家居,房产,教育', type=str, help="文本分类标签")
    parser.add_argument('--train_file', default='data/classification/train.csv', type=str, help="")
    parser.add_argument('--dev_file', default='data/classification/dev.csv', type=str, help="")
    parser.add_argument('--test_file', default='data/classification/test.csv', type=str, help="")

    parser.add_argument("--num_train_epochs", default=3, type=int, help="训练轮次")
    parser.add_argument("--max_seq_length", default=32, type=int, help="最大句子长度")
    parser.add_argument("--batch_size", default=32, type=int, help="训练批次")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累积")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="学习率")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run test on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="是否边训练边验证")
    parser.add_argument("--do_export", action="store_true", help="将模型导出为pb格式.")

    parser.add_argument("--logging_steps", default=1000, type=int, help="训练时每隔几步验证一次")
    parser.add_argument("--saving_steps", default=1000, type=int, help="训练时每隔几步保存一次")
    parser.add_argument("--random_seed", default=42, type=int, help="随机种子")
    parser.add_argument("--threads", default=8, type=int, help="数据处理进程数")
    parser.add_argument("--max_checkpoints", default=1, type=int, help="模型保存最大数量，默认只保存一个")
    parser.add_argument("--single_device", action="store_true", help="是否只使用一个device，默认使用所有的device训练")
    parser.add_argument("--use_xla", action="store_true", help="是否使用XLA加速")
    parser.add_argument(
        "--mixed_precision", action="store_true",
        help="混合精度训练，tf下测试需要同时使用xla才有加速效果，但是开始编译很慢")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not args.single_device:
        args.batch_size = args.batch_size * len(devices())

    args.labels = args.labels.split(',')
    return args


def create_dataset(set_type, tokenizer, args):
    filename_map = {
        'train': args.train_file, 'dev': args.dev_file, 'test': args.test_file
    }
    examples = []
    datas = pd.read_csv(filename_map[set_type], encoding='utf-8', sep='\t').values.tolist()
    for data in datas:
        examples.append(InputExample(
            guid=0, text_a=data[1], label=data[0]
        ))
    features = convert_examples_to_features(
        examples, tokenizer,
        max_length=args.max_seq_length, set_type=set_type,
        label_list=args.labels, threads=args.threads)
    dataset = Dataset(features,
                      is_training=bool(set_type == 'train'),
                      batch_size=args.batch_size,
                      drop_last=bool(set_type == 'train'),
                      buffer_size=len(features),
                      max_length=args.max_seq_length)
    dataset.format_as(['input_ids', 'attention_mask', 'token_type_ids', 'label_ids'])
    return dataset


def get_model_fn(config, args):
    def model_fn(inputs, is_training):
        model = SequenceClassification(
            model_type=args.model_type, config=config,
            num_classes=len(args.labels), is_training=is_training,
            **inputs)

        outputs = {'outputs': {'logits': model.logits, 'label_ids': inputs['label_ids']}}
        if model.loss is not None:
            loss = model.loss / args.gradient_accumulation_steps
            outputs['loss'] = loss
        return outputs

    return model_fn


def get_serving_fn(config, args):
    def serving_fn():
        input_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='input_ids')
        attention_mask = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='attention_mask')
        token_type_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='token_type_ids')
        model = SequenceClassification(
            model_type=args.model_type, config=config,
            num_classes=len(args.labels), is_training=False,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        outputs = {'logits': model.logits}
        return inputs, outputs

    return serving_fn


def metric_fn(outputs: Dict) -> Dict:
    """
    这里定义评估函数
    :param outputs: trainer evaluate 返回的预测结果，model fn的outputs包含哪些字段就会有哪些字段
    :return: 需要返回字典结果
    """
    predictions = np.argmax(outputs['logits'], -1)
    score = accuracy_score(outputs['label_ids'], predictions)
    return {'accuracy': score}


def main():
    args = create_args()
    set_seed(args.random_seed)

    config = CONFIGS[args.model_type].from_pretrained(
        args.model_dir if args.config_path is None else args.config_path)

    tokenizer = TOKENIZERS[args.model_type].from_pretrained(
        args.model_dir if args.vocab_path is None else args.vocab_path, do_lower_case=True)

    train_dataset, dev_dataset, predict_dataset = None, None, None
    if args.do_train:
        train_dataset = create_dataset('train', tokenizer, args)

    if args.do_eval:
        dev_dataset = create_dataset('dev', tokenizer, args)

    if args.do_predict:
        predict_dataset = create_dataset('test', tokenizer, args)

    output_types, output_shapes = (train_dataset or dev_dataset or predict_dataset).output_types_and_shapes()
    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_types=output_types,
        output_shapes=output_shapes,
        metric_fn=metric_fn,
        use_xla=args.use_xla,
        optimizer_type=args.optimizer_type,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_checkpoints=1,
        max_grad=1.0,
        warmup_proportion=args.warmup_proportion,
        mixed_precision=args.mixed_precision,
        single_device=args.single_device,
        logging=True
    )
    trainer.build_model(model_fn=get_model_fn(config, args))
    if args.do_train and train_dataset is not None:
        trainer.compile()
        trainer.from_pretrained(
            args.model_dir if args.pretrained_checkpoint_path is None else args.pretrained_checkpoint_path)

        trainer.train(
            output_dir=args.output_dir,
            evaluate_during_training=args.evaluate_during_training,
            logging_steps=args.logging_steps,
            saving_steps=args.saving_steps,
            greater_is_better=True, metric_for_best_model='accuracy')
        config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval and dev_dataset is not None:
        trainer.from_pretrained(args.output_dir)
        eval_outputs = trainer.evaluate()
        print(json.dumps(
            eval_outputs, ensure_ascii=False, indent=4
        ))

    if args.do_predict and predict_dataset is not None:
        trainer.from_pretrained(args.output_dir)
        outputs = trainer.predict('test', ['logits'], dataset=predict_dataset)
        label_ids = np.argmax(outputs['logits'], axis=-1)
        labels = list(map(lambda x: args.labels[x], label_ids))
        open(
            os.path.join(args.output_dir, 'prediction.txt'), 'w', encoding='utf-8'
        ).write("\n".join(labels))

    if args.do_export:
        trainer.export(
            get_serving_fn(config, args),
            args.output_dir,
            args.export_dir
        )


if __name__ == '__main__':
    main()
