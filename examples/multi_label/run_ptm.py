# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: run_ptm.py
@date: 2020/09/21
"""
import os
import platform
import argparse
import tensorflow.compat.v1 as tf
from textToy.data.classification import (
    InputExample, convert_examples_to_features,
    create_dataset_by_gen, return_types_and_shapes)
from textToy import (Trainer, MultiLabelClassification,
                     CONFIGS, TOKENIZERS,
                     set_seed, ProgressBar, device_count)
from tqdm import tqdm
import json
from textToy.optimizer import create_optimizer
import numpy as np
from textToy.metric.multi_label import multi_label_metric

if platform.system() == 'Windows':
    bar_fn = ProgressBar
else:
    bar_fn = tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser()

parser.add_argument('--model_type', default='bert', type=str, help="模型类型")
parser.add_argument('--data_dir', default='data/multi_label', type=str, help="数据地址")
parser.add_argument('--model_dir', default='bert_base', type=str, help="模型地址")
parser.add_argument('--output_dir', default='ckpt/multi_label', type=str, help="输出文件夹")
parser.add_argument('--batch_size', default=32, type=int, help="批次大小")
parser.add_argument('--max_seq_length', default=128, type=int, help="最大句子长度")
parser.add_argument('--learning_rate', default=2e-5, type=float, help="学习率")
parser.add_argument('--random_seed', default=42, type=int, help="随机种子")
parser.add_argument('--threads', default=8, type=int, help="数据处理线程数")
parser.add_argument('--epochs', default=4, type=int, help="epochs")
parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="梯度累积步数")
parser.add_argument('--use_xla', action="store_true", help="是否开启xla加速训练")
parser.add_argument('--use_torch_mode', action="store_true", help="是否使用torch的backward训练模式，开启之后可以使用梯度累积")

args = parser.parse_args()

if not args.use_torch_mode and args.gradient_accumulation_steps > 1:
    raise ValueError("if you want to use gradient accumulation, please consume use_torch_mode is True.")

args.batch_size = args.batch_size * device_count()

labels = ['DV%d' % (i + 1) for i in range(20)]
threshold = [0.5] * len(labels)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def create_examples(filename):
    examples = []

    with open(filename, 'r', encoding='utf-8') as f:
        for doc in f:
            doc = json.loads(doc)
            for i, sentence in enumerate(doc):
                if sentence['sentence'].strip():
                    examples.append(InputExample(
                        guid=str(i),
                        text_a=sentence['sentence'],
                        label=sentence['labels']
                    ))

    return examples


def load_dataset(set_type, tokenizer):
    examples = create_examples(os.path.join(args.data_dir, set_type + '.json'))

    features = convert_examples_to_features(examples, tokenizer,
                                            max_length=args.max_seq_length, set_type=set_type,
                                            label_list=labels, is_multi_label=True,
                                            threads=args.threads)
    dataset, steps_one_epoch = create_dataset_by_gen(features, args.batch_size, set_type, is_multi_label=True)
    return dataset, steps_one_epoch


def convert_to_one_hot(probs, thresholds):
    if not isinstance(thresholds, list):
        thresholds = [thresholds] * len(probs)
    one_hot = []
    for p, t in zip(probs, thresholds):
        one_hot.append(1 if p > t else 0)
    return one_hot


def predict(trainer, steps, set_type='test'):
    trainer.init_iterator(set_type)
    predictions = None
    output_label_ids = None
    for _ in tqdm(range(steps)):
        if set_type == 'dev':
            loss, pred, label_id = trainer.eval_step()
        else:
            pred, label_id = trainer.test_step()
        if predictions is None:
            predictions = pred
            output_label_ids = label_id
        else:
            predictions = np.append(predictions, pred, 0)
            output_label_ids = np.append(output_label_ids, label_id, 0)
    one_hot = []
    for prediction in predictions:
        one_hot.append(convert_to_one_hot(prediction, threshold))
    return output_label_ids, one_hot


def get_model_fn(model_type, config, num_classes):
    def model_fn(inputs, is_training):
        model = MultiLabelClassification(
            model_type=model_type, config=config, num_classes=num_classes,
            is_training=is_training,
            **inputs
        )
        outputs = [model.predictions, inputs['label_ids']]
        loss = model.loss / args.gradient_accumulation_steps
        return {'loss': loss, 'outputs': outputs}

    return model_fn


def main():
    set_seed(args.random_seed)
    config = CONFIGS[args.model_type].from_pretrained(args.model_dir)
    tokenizer = TOKENIZERS[args.model_type].from_pretrained(args.model_dir, do_lower_case=True)

    train_dataset, num_train_batch = load_dataset('train', tokenizer)
    test_dataset, num_test_batch = load_dataset('test', tokenizer)

    output_types, output_shapes = return_types_and_shapes(for_trainer=True, is_multi_label=True)

    trainer = Trainer(
        args.model_type, output_types, output_shapes, device='gpu', use_xla=args.use_xla, use_torch_mode=args.use_torch_mode
    )

    trainer.build_model(get_model_fn(args.model_type, config, len(labels)))

    t_total = num_train_batch * args.epochs // args.gradient_accumulation_steps

    train_op = create_optimizer(
        init_lr=args.learning_rate,
        gradients=trainer.gradients,
        variables=trainer.variables,
        num_train_steps=t_total,
        num_warmup_steps=t_total * 0.1)

    trainer.compile(
        train_op, max_checkpoints=1)

    trainer.build_handle(train_dataset, 'train')
    trainer.build_handle(test_dataset, 'test')

    trainer.from_pretrained(args.model_dir)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num epochs = {}".format(args.epochs))
    tf.logging.info("  batch size = {}".format(args.batch_size))
    tf.logging.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    tf.logging.info("  Total train batch size (accumulation) = {}".format(args.batch_size * args.gradient_accumulation_steps))
    tf.logging.info("  optimizer steps = %d", t_total)
    tf.logging.info("  Num devices = {}".format(trainer.num_devices))
    tf.logging.info("  Num params = {}".format(trainer.num_params))

    best_score = 0.
    for epoch in range(args.epochs):
        epoch_iter = bar_fn(range(num_train_batch), desc='epoch {} '.format(epoch + 1))
        for step in epoch_iter:
            if args.use_torch_mode:
                train_loss = trainer.backward()
            else:
                train_loss = trainer.train_step()
            epoch_iter.set_description(desc='epoch {} ,loss {:.4f}'.format(epoch + 1, train_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.use_torch_mode:
                    trainer.train_step()
                    trainer.zero_grad()

        y_true, y_pred = predict(trainer, num_test_batch, 'test')
        score = multi_label_metric(y_true, y_pred, label_list=labels)['dict_result']['micro macro avg']['f1-score']
        if score > best_score:
            best_score = score
            trainer.save_pretrained(args.output_dir)
            config.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        tf.logging.info("***** eval results *****")
        tf.logging.info(" global step : {}".format(trainer.global_step))
        tf.logging.info(" eval score : {:.4f}".format(score))
        tf.logging.info(" best score : {:.4f}".format(best_score))

    tf.logging.info("***** Running Test *****")
    trainer.from_pretrained(args.output_dir)
    y_true, y_pred = predict(trainer, num_test_batch, 'test')
    report = multi_label_metric(y_true, y_pred, label_list=labels)['string_result']
    open(os.path.join(args.output_dir, 'result.txt'), 'w', encoding='utf-8').write(report)
    tf.logging.info("***** test results *****")
    report = report.split('\n')
    for r in report:
        tf.logging.info(r)


if __name__ == '__main__':
    main()
