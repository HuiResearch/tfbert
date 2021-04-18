# -*- coding:utf-8 -*-
# @FileName  :run_ner.py
# @Time      :2021/2/8 13:44
# @Author    :huanghui
import tensorflow.compat.v1 as tf
from tfbert import (
    Trainer, Dataset,
    TokenClassification,
    CONFIGS, TOKENIZERS, devices, set_seed)
from tfbert.data.ner import convert_examples_to_features, InputExample
from tfbert.metric.ner import ner_metric
import os
import argparse


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
    parser.add_argument('--output_dir', default='output/ner', type=str, help="")
    parser.add_argument('--export_dir', default='output/ner/pb', type=str, help="")

    parser.add_argument('--labels', default='O,B-PER,I-PER,B-ORG,I-ORG,B-LOC,I-LOC', type=str, help="文本分类标签")
    parser.add_argument('--train_file', default='data/ner/train.txt', type=str, help="")
    parser.add_argument('--dev_file', default='data/ner/dev.txt', type=str, help="")
    parser.add_argument('--test_file', default='data/ner/test.txt', type=str, help="")

    parser.add_argument("--num_train_epochs", default=3, type=int, help="训练轮次")
    parser.add_argument("--max_seq_length", default=180, type=int, help="最大句子长度")
    parser.add_argument("--batch_size", default=16, type=int, help="训练批次")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累积")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="学习率")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--add_crf", action="store_true", help="是否增加crf层.")

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


def create_examples(filename):
    examples = []
    words = []
    tags = []
    with open(filename, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            if line:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)

            elif words and tags:
                examples.append(
                    InputExample(
                        guid=str(len(examples)),
                        words=words,
                        tags=tags
                    )
                )
                words = []
                tags = []
    return examples


def create_dataset(set_type, tokenizer, args, return_examples=False):
    filename_map = {
        'train': args.train_file, 'dev': args.dev_file, 'test': args.test_file
    }
    examples = create_examples(filename_map[set_type])
    features = convert_examples_to_features(
        examples, tokenizer, args.max_seq_length, args.labels, set_type,
        pad_token_label_id=0,
        threads=args.threads
    )

    dataset = Dataset(features,
                      is_training=bool(set_type == 'train'),
                      batch_size=args.batch_size,
                      drop_last=bool(set_type == 'train'),
                      buffer_size=len(features),
                      max_length=args.max_seq_length)
    dataset.format_as(['input_ids', 'input_mask', 'token_type_ids', 'label_ids'])
    if not return_examples:
        return dataset
    return dataset, examples, features


def get_model_fn(config, args):
    def model_fn(inputs, is_training):
        model = TokenClassification(
            model_type=args.model_type, config=config,
            num_classes=len(args.labels), is_training=is_training,
            add_crf=args.add_crf,
            **inputs)

        outputs = {
            'outputs': {
                'predictions': model.predictions,
                'label_ids': inputs['label_ids'],
                'input_mask': inputs['input_mask']}}
        if model.loss is not None:
            loss = model.loss / args.gradient_accumulation_steps
            outputs['loss'] = loss
        return outputs

    return model_fn


def get_serving_fn(config, args):
    def serving_fn():
        input_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='input_ids')
        input_mask = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='input_mask')
        token_type_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='token_type_ids')
        model = TokenClassification(
            model_type=args.model_type, config=config,
            num_classes=len(args.labels), is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            add_crf=args.add_crf
        )
        inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'token_type_ids': token_type_ids}
        outputs = {'predictions': model.predictions}
        return inputs, outputs

    return serving_fn


def get_post_process_fn(labels):
    def post_process_fn(outputs):
        id2label = dict(zip(range(len(labels)), labels))

        def convert_to_label(ids):
            return list(map(lambda x: id2label[x], ids))

        y_pred, y_true = [], []
        for prediction, output_label_id, mask in zip(
                outputs['predictions'], outputs['label_ids'], outputs['input_mask']):
            pred_ids = []
            true_ids = []
            for p, t, m in zip(prediction, output_label_id, mask):
                # 去除填充位置
                if m == 1:
                    pred_ids.append(p)
                    true_ids.append(t)
                else:
                    break
            pred_ids = pred_ids[1: -1]  # 去掉cls、sep位置
            true_ids = true_ids[1: -1]
            y_pred.append(convert_to_label(pred_ids))
            y_true.append(convert_to_label(true_ids))
        return {'predict_tags': y_pred, 'source_tags': y_true}

    return post_process_fn


def get_metric_fn():
    def metric_fn(outputs):
        result = ner_metric(
            y_true=outputs['source_tags'], y_pred=outputs['predict_tags'], dict_report=True
        )
        return {
            'macro-f1': result[1]['macro avg']['f1-score'],
            'micro-f1': result[1]['micro avg']['f1-score'],
            'report': result[0]
        }

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
        dev_dataset = create_dataset('dev', tokenizer, args)

    if args.do_predict:
        predict_dataset, predict_examples, predict_features = create_dataset(
            'test', tokenizer, args, return_examples=True)

    output_types, output_shapes = (train_dataset or dev_dataset or predict_dataset).output_types_and_shapes()
    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_types=output_types,
        output_shapes=output_shapes,
        metric_fn=get_metric_fn(),
        post_process_fn=get_post_process_fn(args.labels),
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
            greater_is_better=True, metric_for_best_model='macro-f1')
        config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval and dev_dataset is not None:
        trainer.from_pretrained(args.output_dir)
        eval_outputs = trainer.evaluate()
        print(eval_outputs['report'])

    if args.do_predict and predict_dataset is not None:
        trainer.from_pretrained(args.output_dir)
        outputs = trainer.predict('test', dataset=predict_dataset)
        with open(
                os.path.join(args.output_dir, 'prediction.txt'), 'w', encoding='utf-8'
        ) as f:
            for example, feature, pred_tags in zip(
                    predict_examples, predict_features,
                    outputs['predict_tags']):
                tags = ['O'] * len(example.words)
                for i in range(len(pred_tags)):
                    tags[feature.tok_to_orig_index[i]] = pred_tags[i]
                for w, t in zip(example.words, tags):
                    f.write(f"{w}\t{t}\n")

    if args.do_export:
        trainer.export(
            get_serving_fn(config, args),
            args.output_dir,
            args.export_dir
        )


if __name__ == '__main__':
    main()
