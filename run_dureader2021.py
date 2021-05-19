# -*- coding: UTF-8 -*-
__author__ = 'huanghui'
__date__ = '2021/5/17 23:23'
__project__ = 'tfbert'

import os
import json
import argparse
import tensorflow.compat.v1 as tf
from tfbert import (
    Dataset, set_seed, QuestionAnswering,
    CONFIGS, TOKENIZERS, devices, Trainer)
from tfbert.data.mrc import (
    convert_examples_to_features, MrcProcessor,
    compute_predictions_logits, SquadResult, SquadExample, SquadFeatures)
from tfbert.metric.mrc import metric
from typing import Dict, List


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

    parser.add_argument('--train_file', default='data/dureader2021/train.json', type=str, help="")
    parser.add_argument('--dev_file', default='data/dureader2021/dev.json', type=str, help="")
    parser.add_argument('--test_file', default='data/dureader2021/test1.json', type=str, help="")

    parser.add_argument("--num_train_epochs", default=2, type=int, help="训练轮次")
    parser.add_argument("--batch_size", default=8, type=int, help="训练批次")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累积")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="学习率")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument("--max_seq_length", default=384, type=int, help="最大句子长度")
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=32,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument(
        "--n_best_size",
        default=10,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=384,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run test on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="是否边训练边验证")
    parser.add_argument("--do_export", action="store_true", help="将模型导出为pb格式.")

    parser.add_argument("--logging_steps", default=1000, type=int, help="训练时每隔几步验证一次")
    parser.add_argument("--saving_steps", default=1000, type=int, help="训练时每隔几步保存一次")
    parser.add_argument("--random_seed", default=42, type=int, help="随机种子")
    parser.add_argument("--threads", default=1, type=int, help="数据处理进程数")
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

    return args


def create_dataset(set_type, tokenizer, args, return_examples=False):
    processor = MrcProcessor()
    if set_type == 'train':
        examples = processor.get_train_examples(args.train_file)
    elif set_type == 'dev':
        examples = processor.get_dev_examples(args.dev_file)
    else:
        examples = processor.get_test_examples(args.test_file)

    features = convert_examples_to_features(
        examples, tokenizer,
        args.max_seq_length,
        args.doc_stride,
        args.max_query_length,
        set_type=set_type,
        threads=args.threads
    )
    dataset = Dataset(features,
                      is_training=bool(set_type == 'train'),
                      batch_size=args.batch_size,
                      drop_last=bool(set_type == 'train'),
                      buffer_size=len(features),
                      max_length=args.max_seq_length)
    dataset.format_as(['input_ids', 'attention_mask', 'token_type_ids', 'start_position', 'end_position'])
    if return_examples:
        return dataset, examples, features
    return dataset


def get_model_fn(config, args):
    def model_fn(inputs, is_training):
        model = QuestionAnswering(
            model_type=args.model_type, config=config,
            is_training=is_training,
            **inputs)

        outputs = {'outputs': {'start_logits': model.start_logits, 'end_logits': model.end_logits}}
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
        model = QuestionAnswering(
            model_type=args.model_type, config=config,
            is_training=False,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        outputs = {'start_logits': model.start_logits, 'end_logits': model.end_logits}
        return inputs, outputs

    return serving_fn


def get_post_process_fn(args, tokenizer, examples: List[SquadExample], features: List[SquadFeatures]):
    def post_process_fn(outputs: Dict):
        results = []
        for i in range(len(features)):
            start_logits = outputs['start_logits'][i].tolist()
            end_logits = outputs['end_logits'][i].tolist()
            unique_id = features[i].unique_id
            results.append(
                SquadResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits)
            )
        predictions, _, _ = compute_predictions_logits(
            all_examples=examples, all_features=features, all_results=results,
            n_best_size=args.n_best_size, max_answer_length=args.max_answer_length,
            do_lower_case=True, output_prediction_file=None, output_nbest_file=None,
            output_null_log_odds_file=None, verbose_logging=False,
            version_2_with_negative=args.version_2_with_negative,
            null_score_diff_threshold=args.null_score_diff_threshold,
            tokenizer=tokenizer,
            empty_answer='no answer'
        )
        return {'predictions': predictions}

    return post_process_fn


def get_metric_fn(gold_file):
    def metric_fn(outputs):
        result = metric(
            predictions=outputs['predictions'], gold_file=gold_file, dict_report=True
        )
        # 这里的result元组，第一个为字符串类型的评估结果，第二个为字典结果
        return result[1]

    return metric_fn


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
        dev_dataset, dev_examples, dev_features = create_dataset('dev', tokenizer, args, return_examples=True)

    if args.do_predict:
        predict_dataset, predict_examples, predict_features = create_dataset(
            'test', tokenizer, args, return_examples=True)
    output_types, output_shapes = (train_dataset or dev_dataset or predict_dataset).output_types_and_shapes()
    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_types=output_types,
        output_shapes=output_shapes,
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

        # 训练过程中结果后处理需要传入的是验证examples
        trainer.train(
            output_dir=args.output_dir,
            evaluate_during_training=args.evaluate_during_training,
            metric_fn=get_metric_fn(args.dev_file),
            post_process_fn=get_post_process_fn(args, tokenizer, dev_examples,
                                                dev_features) if args.evaluate_during_training else None,
            logging_steps=args.logging_steps,
            saving_steps=args.saving_steps,
            greater_is_better=True,
            metric_for_best_model='f1')
        config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval and dev_dataset is not None:
        trainer.from_pretrained(args.output_dir)
        # 验证过程中结果后处理需要传入的是验证examples
        eval_outputs = trainer.evaluate(
            eval_dataset=dev_dataset,
            eval_steps=0,
            metric_fn=get_metric_fn(args.dev_file),  # 标准文件用验证集文件
            post_process_fn=get_post_process_fn(args, tokenizer, dev_examples,
                                                dev_features))
        tf.logging.info("***** eval results *****")
        print(eval_outputs)

    if args.do_predict and predict_dataset is not None:
        trainer.from_pretrained(args.output_dir)
        # 预测过程中结果后处理需要传入的是测试集examples
        outputs = trainer.predict(
            'test', dataset=predict_dataset,
            post_process_fn=get_post_process_fn(args, tokenizer, predict_examples, predict_features))

        # 去除自定义的post process fn的结果，存进json文件
        open(os.path.join(args.output_dir, 'predictions.json'), 'w', encoding='utf-8').write(
            json.dumps(outputs['predictions'], ensure_ascii=False, indent=4)
        )
    if args.do_export:
        trainer.export(
            get_serving_fn(config, args),
            args.output_dir,
            args.export_dir
        )


if __name__ == '__main__':
    main()
