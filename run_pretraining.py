# -*- coding: UTF-8 -*-
# author : 'huanghui'
#  date  : '2021/5/21 9:14'
# project: 'tfbert'
"""
使用 gen dataset 实现动态mask，但是数据量大的话可能搞不定，需要使用tfrecord。
"""

import json
import os
import argparse
import random
import tensorflow.compat.v1 as tf
from tfbert import (
    Trainer, MaskedLM,
    CONFIGS, TOKENIZERS, devices, set_seed,
    compute_types, compute_shapes, process_dataset)
from tfbert.data.pretrain import create_masked_lm_predictions, convert_to_unicode
from tfbert.tokenizer.tokenization_base import PTMTokenizer
from typing import Dict, List
from sklearn.metrics import accuracy_score


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
    parser.add_argument('--output_dir', default='output/pretrain', type=str, help="")
    parser.add_argument('--export_dir', default='output/pretrain/pb', type=str, help="")

    parser.add_argument('--train_dir', default='data/pretrain/train', type=str, help="训练文件所在文件夹")
    parser.add_argument('--dev_dir', default='data/pretrain/dev', type=str, help="验证文件所在文件夹")

    parser.add_argument("--num_train_epochs", default=10, type=int, help="训练轮次")
    parser.add_argument("--max_seq_length", default=128, type=int, help="最大句子长度")
    parser.add_argument("--batch_size", default=64, type=int, help="训练批次")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累积")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")

    parser.add_argument("--masked_lm_prob", default=0.15, type=float, help="mask 概率.")
    parser.add_argument("--max_predictions_per_seq", default=20, type=int, help="最大mask数量.")
    parser.add_argument("--ngram", default=4, type=int, help="ngram mask 最大个数.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="是否边训练边验证")
    parser.add_argument("--do_export", action="store_true", help="将模型导出为pb格式.")
    parser.add_argument("--do_whole_word_mask", action="store_true", help="全词mask.")

    parser.add_argument("--logging_steps", default=1000, type=int, help="训练时每隔几步验证一次")
    parser.add_argument("--saving_steps", default=1000, type=int, help="训练时每隔几步保存一次")
    parser.add_argument("--random_seed", default=42, type=int, help="随机种子")
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

    def find_files(dir_or_file):
        if os.path.isdir(dir_or_file):
            files = os.listdir(dir_or_file)
            files = [os.path.join(dir_or_file, file_) for file_ in files]
        elif isinstance(dir_or_file, str):
            files = [dir_or_file]
        else:
            files = []
        return files

    args.train_files = find_files(args.train_dir)
    args.dev_files = find_files(args.dev_dir)

    if len(args.dev_files) == 0:
        args.do_eval = False
        args.evaluate_during_training = False

    if len(args.train_files) == 0 and args.do_train:
        args.do_train = False
        tf.logging.warn("If you need to perform training, please ensure that the training file is not empty")
    return args


def create_dataset(args, input_files, tokenizer: PTMTokenizer, set_type):
    if not isinstance(input_files, List):
        input_files = [input_files]
    all_tokens = []
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = convert_to_unicode(line)
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                tokens = tokenizer.tokenize(line)
                tokens = tokens[:args.max_seq_length - 2]
                tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

                all_tokens.append(tokens)
    # 打乱
    random.shuffle(all_tokens)

    # 定义生成器，提供动态mask
    def dynamic_mask_gen():
        for tokens in all_tokens:
            output_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                tokens, args.masked_lm_prob,
                args.max_predictions_per_seq,
                list(tokenizer.vocab.keys()),
                do_whole_word_mask=args.do_whole_word_mask,
                favor_shorter_ngram=True,
                ngram=args.ngram
            )

            encoded = tokenizer.encode_plus(
                output_tokens, add_special_tokens=False, padding="max_length",
                truncation=True, max_length=args.max_seq_length
            )
            masked_lm_positions = list(masked_lm_positions)
            masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
            masked_lm_weights = [1.0] * len(masked_lm_ids)

            while len(masked_lm_positions) < args.max_predictions_per_seq:
                masked_lm_positions.append(0)
                masked_lm_ids.append(0)
                masked_lm_weights.append(0.0)
            encoded.update(
                {'masked_lm_ids': masked_lm_ids,
                 'masked_lm_weights': masked_lm_weights,
                 'masked_lm_positions': masked_lm_positions}
            )
            yield encoded

    sample_example = {
        'input_ids': [0], 'token_type_ids': [0], 'attention_mask': [0],
        'masked_lm_ids': [0], 'masked_lm_weights': [0.0], 'masked_lm_positions': [0]
    }
    types = compute_types(sample_example)
    shapes = compute_shapes(sample_example)
    dataset = tf.data.Dataset.from_generator(
        dynamic_mask_gen, types, shapes
    )
    dataset, steps = process_dataset(
        dataset, args.batch_size, len(all_tokens), set_type, buffer_size=100)
    return dataset, steps


def get_model_fn(config, args):
    def model_fn(inputs, is_training):
        model = MaskedLM(
            model_type=args.model_type,
            config=config,
            is_training=is_training,
            **inputs)

        masked_lm_ids = tf.reshape(inputs['masked_lm_ids'], [-1])
        masked_lm_log_probs = tf.reshape(model.prediction_scores,
                                         [-1, model.prediction_scores.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_weights = tf.reshape(inputs['masked_lm_weights'], [-1])

        outputs = {'outputs': {
            'masked_lm_predictions': masked_lm_predictions,
            'masked_lm_ids': masked_lm_ids,
            'masked_lm_weights': masked_lm_weights
        }}
        if model.loss is not None:
            loss = model.loss / args.gradient_accumulation_steps
            outputs['loss'] = loss
        return outputs

    return model_fn


def metric_fn(outputs: Dict) -> Dict:
    """
    这里定义评估函数
    :param outputs: trainer evaluate 返回的预测结果，model fn的outputs包含哪些字段就会有哪些字段
    :return: 需要返回字典结果
    """
    score = accuracy_score(outputs['masked_lm_ids'], outputs['masked_lm_predictions'],
                           sample_weight=outputs['masked_lm_weights'])
    return {'accuracy': score}


def main():
    args = create_args()
    set_seed(args.random_seed)

    config = CONFIGS[args.model_type].from_pretrained(
        args.model_dir if args.config_path is None else args.config_path)

    tokenizer = TOKENIZERS[args.model_type].from_pretrained(
        args.model_dir if args.vocab_path is None else args.vocab_path, do_lower_case=True)

    # tf 自带的dataset不知道怎么自动得到一轮需要步数
    # 因此提前算出来传入trainer
    train_dataset, train_steps, dev_dataset, dev_steps = None, 0, None, 0
    if args.do_train:
        train_dataset, train_steps = create_dataset(args, args.train_files, tokenizer, 'train')

    if args.do_eval:
        dev_dataset, dev_steps = create_dataset(args, args.dev_files, tokenizer, 'dev')

    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
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
        # 训练阶段需要先compile优化器才能初始化权重
        # 因为adam也是具备参数的
        trainer.compile()
    trainer.from_pretrained(
        args.model_dir if args.pretrained_checkpoint_path is None else args.pretrained_checkpoint_path)
    if args.do_train and train_dataset is not None:
        trainer.compile()

        trainer.train(
            output_dir=args.output_dir,
            train_steps=train_steps,  # 这个是一轮的步数
            eval_steps=dev_steps,
            evaluate_during_training=args.evaluate_during_training,
            logging_steps=args.logging_steps,
            saving_steps=args.saving_steps,
            greater_is_better=True,
            load_best_model=True,
            metric_for_best_model='accuracy')
        config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval and dev_dataset is not None:
        eval_outputs = trainer.evaluate(eval_steps=dev_steps)
        print(json.dumps(
            eval_outputs, ensure_ascii=False, indent=4
        ))


if __name__ == '__main__':
    main()
