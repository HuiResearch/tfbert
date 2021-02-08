# -*- coding:utf-8 -*-
# @FileName  :run_ner.py
# @Time      :2021/2/8 13:44
# @Author    :huanghui
import platform
import tensorflow.compat.v1 as tf
from tfbert.data.ner import (
    InputExample, convert_examples_to_features,
    create_dataset_by_gen, return_types_and_shapes)
from tfbert import (TokenClassification,
                    CONFIGS, TOKENIZERS,
                    set_seed, ProgressBar,
                    create_optimizer,
                    devices, Trainer)
from tfbert.metric.ner import ner_metric
from tqdm import tqdm
import os
import argparse

if platform.system() == 'Windows':
    bar_fn = ProgressBar  # win10下我使用tqdm老换行，所以自己写了一个
else:
    bar_fn = tqdm  # linux就用tqdm


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
    parser.add_argument("--warmup_proportion", default=0.1, type=int,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--weight_decay", default=0.01, type=int, help="Weight decay if we apply some.")
    parser.add_argument("--add_crf", action="store_true", help="是否增加crf层.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="是否边训练边验证")
    parser.add_argument("--do_export", action="store_true", help="将模型导出为pb格式.")

    parser.add_argument("--logging_steps", default=1000, type=int, help="训练时每隔几步验证一次")
    parser.add_argument("--save_steps", default=1000, type=int, help="训练时每隔几步保存一次")
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


def load_dataset(set_type, tokenizer, args, return_examples=False):
    filename_map = {
        'train': args.train_file, 'dev': args.dev_file, 'test': args.test_file
    }
    examples = create_examples(filename_map[set_type])
    features = convert_examples_to_features(
        examples, tokenizer, args.max_seq_length, args.labels, set_type,
        pad_token_label_id=0,
        threads=args.threads
    )

    dataset, steps_one_epoch = create_dataset_by_gen(features, args.batch_size, set_type)
    if not return_examples:
        return dataset, steps_one_epoch
    return dataset, steps_one_epoch, examples, features


def get_model_fn(config, args):
    def model_fn(inputs, is_training):
        model = TokenClassification(
            model_type=args.model_type, config=config,
            num_classes=len(args.labels), is_training=is_training,
            add_crf=args.add_crf,
            **inputs)
        loss = model.loss / args.gradient_accumulation_steps
        return {
            'loss': loss,
            'outputs': {
                'predictions': model.predictions,
                'label_ids': inputs['label_ids'],
                'input_mask': inputs['input_mask']}}

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


def evaluate(trainer: Trainer, labels, set_type='dev', steps=None):
    id2label = dict(zip(range(len(labels)), labels))

    def convert_to_label(ids):
        return list(map(lambda x: id2label[x], ids))

    outputs = trainer.predict(set_type, ['predictions', 'label_ids', 'input_mask'], steps)
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
    result = ner_metric(
        y_true=y_true, y_pred=y_pred, dict_report=True
    )
    return result


def train(config, tokenizer, args):
    train_dataset, num_train_batch = load_dataset('train', tokenizer, args)

    if args.evaluate_during_training:
        dev_dataset, num_dev_batch = load_dataset('dev', tokenizer, args)
    else:
        dev_dataset, num_dev_batch = None, None

    t_total = num_train_batch * args.num_train_epochs // args.gradient_accumulation_steps

    optimizer = create_optimizer(
        args.learning_rate,
        num_train_steps=t_total,
        num_warmup_steps=t_total * args.warmup_proportion,
        optimizer_type=args.optimizer_type,
        mixed_precision=args.mixed_precision
    )
    input_types, input_shapes = return_types_and_shapes(for_trainer=True)
    # 初始化trainer
    trainer = Trainer(
        input_types=input_types,
        input_shapes=input_shapes,
        optimizer=optimizer,  # 因为使用混合精度训练需要使用rewrite过的优化器计算梯度，所以需要先传入，如果不使用就可以在compile传入
        use_xla=args.use_xla,
        mixed_precision=args.mixed_precision,
        single_device=args.single_device
    )

    # 构建模型
    trainer.build_model(get_model_fn(config, args))
    # 配置trainer优化器
    trainer.compile(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad=1.,
    )
    trainer.prepare_dataset(train_dataset, 'train')
    if dev_dataset is not None:
        trainer.prepare_dataset(dev_dataset, 'dev')

    # 预训练模型加载预训练参数，若不加载，调用trainer.init_variables()
    trainer.from_pretrained(
        args.model_dir if args.pretrained_checkpoint_path is None else args.pretrained_checkpoint_path)

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
        epoch_iter = bar_fn(range(num_train_batch), desc='epoch {} '.format(epoch + 1))
        for step in epoch_iter:
            train_loss = trainer.train_step()
            epoch_iter.set_description(desc='epoch {} ,loss {:.4f}'.format(epoch + 1, train_loss))

            if args.evaluate_during_training and trainer.global_step_changed and (
                    trainer.global_step % args.logging_steps == 0 or trainer.global_step == t_total):
                eval_result = evaluate(trainer, args.labels, 'dev', num_dev_batch)
                f1_score = eval_result[1]['macro avg']['f1-score']
                if f1_score > best_score:
                    best_score = f1_score
                    trainer.save_pretrained(args.output_dir)
                    config.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                tf.logging.info("***** eval results *****")
                tf.logging.info(" global step : {}".format(trainer.global_step))
                tf.logging.info(" eval score : {:.4f}".format(f1_score))
                tf.logging.info(" best score : {:.4f}".format(best_score))
            if not args.evaluate_during_training and trainer.global_step_changed and (
                    trainer.global_step % args.save_steps == 0 or trainer.global_step == t_total):
                trainer.save_pretrained(args.output_dir)
                config.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
        epoch_iter.close()

    tf.logging.info("***** Finished training *****")
    return trainer  # 返回配置好的trainer给验证测试调用


def main():
    args = create_args()
    set_seed(args.random_seed)

    trainer = None
    config = None
    tokenizer = None
    if args.do_train:
        config = CONFIGS[args.model_type].from_pretrained(
            args.model_dir if args.config_path is None else args.config_path)
        tokenizer = TOKENIZERS[args.model_type].from_pretrained(
            args.model_dir if args.vocab_path is None else args.vocab_path, do_lower_case=True)
        trainer = train(config, tokenizer, args)

    if args.do_eval or args.do_test or args.do_export:
        config = CONFIGS[args.model_type].from_pretrained(args.output_dir)
        tokenizer = TOKENIZERS[args.model_type].from_pretrained(args.output_dir, do_lower_case=True)

    if trainer is None and (args.do_eval or args.do_test or args.do_export):
        input_types, input_shapes = return_types_and_shapes(for_trainer=True)
        trainer = Trainer(
            input_types=input_types,
            input_shapes=input_shapes,
            use_xla=args.use_xla,
            mixed_precision=args.mixed_precision,
            single_device=args.single_device
        )
        trainer.build_model(get_model_fn(config, args))
    if args.do_eval:
        dev_dataset, num_dev_batch = load_dataset('dev', tokenizer, args)
        trainer.prepare_dataset(dev_dataset, 'dev')
        trainer.from_pretrained(args.output_dir)
        eval_result = evaluate(trainer, args.labels, 'dev', num_dev_batch)
        report = eval_result[0].split('\n')
        with open(os.path.join(args.output_dir, "eval_result.txt"), 'w', encoding='utf-8') as w:
            for r in report:
                tf.logging.info(r)
                w.write(r + '\n')

    if args.do_test:
        test_dataset, num_test_batch, test_examples, test_features = load_dataset(
            'test', tokenizer, args, return_examples=True)

        trainer.from_pretrained(args.output_dir)
        trainer.prepare_dataset(test_dataset, 'test')
        outputs = trainer.predict('test', ['predictions', 'input_mask'], num_test_batch)

        with open(
                os.path.join(args.output_dir, 'prediction.txt'), 'w', encoding='utf-8'
        ) as f:
            for example, feature, pred, mask in zip(
                    test_examples, test_features,
                    outputs['predictions'], outputs['input_mask']):
                pred_ids = []
                for p, m in zip(pred, mask):
                    # 去除填充位置
                    if m == 1:
                        pred_ids.append(args.labels[p])
                    else:
                        break
                pred_ids = pred_ids[1:-1]
                tags = ['O'] * len(example.words)
                for i in range(len(pred_ids)):
                    tags[feature.tok_to_orig_index[i]] = pred_ids[i]
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
