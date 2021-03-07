# -*- coding:utf-8 -*-
# @FileName  :run_element_extract.py
# @Time      :2021/2/5 10:27
# @Author    :huanghui
import platform
import tensorflow.compat.v1 as tf
from tfbert.data.classification import (
    InputExample, convert_examples_to_features,
    create_dataset_by_gen, return_types_and_shapes)
from tfbert import (MultiLabelClassification,
                    CONFIGS, TOKENIZERS,
                    set_seed, ProgressBar,
                    devices, Trainer, create_optimizer)
from tfbert.metric.multi_label import multi_label_metric
from tqdm import tqdm
import os
import json
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
    parser.add_argument('--output_dir', default='output/multi_label', type=str, help="")
    parser.add_argument('--export_dir', default='output/multi_label/pb', type=str, help="")

    parser.add_argument('--ay', default='DV', type=str, help="司法要素抽取的案由，DV,LB")
    parser.add_argument('--train_file', default='data/multi_label/train.json', type=str, help="")
    parser.add_argument('--dev_file', default='data/multi_label/test.json', type=str, help="")
    parser.add_argument('--test_file', default='data/multi_label/test.json', type=str, help="")

    parser.add_argument("--num_train_epochs", default=3, type=int, help="训练轮次")
    parser.add_argument("--max_seq_length", default=128, type=int, help="最大句子长度")
    parser.add_argument("--batch_size", default=32, type=int, help="训练批次")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累积")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="学习率")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--threshold", default=0.5, type=float, help="多标签分类判定阈值，每个标签都用这个，有需要可以定义为列表.")

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

    args.labels = [f'{args.ay}{i + 1}' for i in range(20)]
    return args


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


def load_dataset(set_type, tokenizer, args):
    filename_map = {
        'train': args.train_file, 'dev': args.dev_file, 'test': args.test_file
    }
    examples = create_examples(filename_map[set_type])

    features = convert_examples_to_features(examples, tokenizer,
                                            max_length=args.max_seq_length, set_type=set_type,
                                            label_list=args.labels, is_multi_label=True,
                                            threads=args.threads)
    dataset, steps_one_epoch = create_dataset_by_gen(
        features, args.batch_size, set_type, is_multi_label=True)
    return dataset, steps_one_epoch


def convert_to_one_hot(probs, thresholds):
    if not isinstance(thresholds, list):
        thresholds = [thresholds] * len(probs)
    one_hot = []
    for p, t in zip(probs, thresholds):
        one_hot.append(1 if p > t else 0)
    return one_hot


def get_model_fn(config, args):
    def model_fn(inputs, is_training):
        model = MultiLabelClassification(
            model_type=args.model_type,
            config=config,
            num_classes=len(args.labels),
            is_training=is_training,
            **inputs
        )
        loss = model.loss / args.gradient_accumulation_steps
        return {
            'loss': loss,
            'outputs': {'predictions': model.predictions, 'label_ids': inputs['label_ids']}}

    return model_fn


def get_serving_fn(config, args):
    def serving_fn():
        input_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='input_ids')
        input_mask = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='input_mask')
        token_type_ids = tf.placeholder(shape=[None, args.max_seq_length], dtype=tf.int64, name='token_type_ids')
        model = MultiLabelClassification(
            model_type=args.model_type,
            config=config,
            num_classes=len(args.labels),
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids
        )
        inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'token_type_ids': token_type_ids}
        outputs = {'predictions': model.predictions}
        return inputs, outputs

    return serving_fn


def evaluate(args, trainer, set_type='dev', steps=None):
    outputs = trainer.predict(set_type, ['predictions', 'label_ids'], steps)
    one_hot = []
    for prediction in outputs['predictions']:
        one_hot.append(convert_to_one_hot(prediction, args.threshold))
    result = multi_label_metric(outputs['label_ids'], one_hot, args.labels, dict_report=True)
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
    input_types, input_shapes = return_types_and_shapes(
        for_trainer=True, is_multi_label=True)
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
    # 配置trainer训练节点train op
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
                result = evaluate(
                    args, trainer, 'dev', num_dev_batch
                )
                score = result[1]['micro macro avg']['f1-score']
                if score > best_score:
                    best_score = score
                    trainer.save_pretrained(args.output_dir)
                    config.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                tf.logging.info("***** eval results *****")
                tf.logging.info(" global step : {}".format(trainer.global_step))
                tf.logging.info(" eval score : {:.4f}".format(score))
                tf.logging.info(" best score : {:.4f}".format(best_score))
            if not args.evaluate_during_training and trainer.global_step_changed and (
                    trainer.global_step % args.save_steps == 0 or trainer.global_step == t_total):
                trainer.save_pretrained(args.output_dir)
                config.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
        epoch_iter.close()

    tf.logging.info("***** Finished training *****")
    return trainer


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
        input_types, input_shapes = return_types_and_shapes(for_trainer=True, is_multi_label=True)
        trainer = Trainer(
            input_types=input_types,
            input_shapes=input_shapes,
            use_xla=args.use_xla,
            mixed_precision=args.mixed_precision,
            single_device=args.single_device
        )
        trainer.build_model(get_model_fn(config, args))
    if args.do_eval:
        tf.logging.info("***** Running Evaluation *****")
        dev_dataset, num_dev_batch = load_dataset('dev', tokenizer, args)
        trainer.prepare_dataset(dev_dataset, 'dev')
        trainer.from_pretrained(args.output_dir)
        result = evaluate(args, trainer, 'dev', num_dev_batch)
        tf.logging.info("***** eval results *****")
        report = result[0].split('\n')
        with open(os.path.join(args.output_dir, "eval_result.txt"), 'w', encoding='utf-8') as w:
            for r in report:
                tf.logging.info(r)
                w.write(r + '\n')

    if args.do_test:
        test_dataset, num_test_batch = load_dataset('test', tokenizer, args)
        trainer.from_pretrained(args.output_dir)
        trainer.prepare_dataset(test_dataset, 'test')
        outputs = trainer.predict('test', ['predictions'], num_test_batch)
        labels = []
        for prediction in outputs['predictions']:
            one_hot = convert_to_one_hot(prediction, args.threshold)
            labels.append(
                [args.labels[i] for i in range(len(one_hot)) if one_hot[i] == 1]
            )
        with open(
                os.path.join(args.output_dir, 'prediction.json'), 'w', encoding='utf-8'
        ) as w:
            for label in labels:
                w.write(json.dumps(label, ensure_ascii=False) + '\n')

    if args.do_export:
        trainer.export(
            get_serving_fn(config, args),
            args.output_dir,
            args.export_dir
        )


if __name__ == '__main__':
    main()
