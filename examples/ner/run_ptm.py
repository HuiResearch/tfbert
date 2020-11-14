# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: run_ptm.py
@date: 2020/09/21
"""
import os
import platform
import tensorflow.compat.v1 as tf
from textToy.data.ner import (
    InputExample, convert_examples_to_features, create_dataset_by_gen, return_types_and_shapes)
from textToy import (Trainer, TokenClassification,
                     CONFIGS, TOKENIZERS,
                     set_seed, ProgressBar)
from tqdm import tqdm
from textToy.optimizer import create_optimizer
from textToy.metric.ner import ner_report, prf_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if platform.system() == 'Windows':
    bar_fn = ProgressBar
else:
    bar_fn = tqdm

model_type = 'bert'
data_dir = 'data/ner'
bert_dir = "bert_base"
output_dir = 'ckpt/ner'

batch_size = 32
max_seq_length = 180
learning_rate = 2e-5
random_seed = 42
epochs = 3
labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
logging_steps = 1000
threads = 8
add_crf = False
gradient_accumulation_steps = 1  # 梯度累积步数
use_xla = False  # 开启xla加速
use_torch_mode = False  # torch模式训练会先backward，然后再train，这样可以支持梯度累积，但是速度会慢

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not use_torch_mode and gradient_accumulation_steps > 1:
    raise ValueError("if you want to use gradient accumulation, please consume use_torch_mode is True.")


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


def load_dataset(set_type, tokenizer):
    examples = create_examples(os.path.join(data_dir, set_type + '.txt'))
    features = convert_examples_to_features(
        examples, tokenizer, max_seq_length, labels, set_type, pad_token_label_id=0,
        use_multi_threads=True, threads=threads
    )

    dataset, steps_one_epoch = create_dataset_by_gen(features, batch_size, set_type)
    return dataset, steps_one_epoch


def predict(trainer, steps, set_type='dev'):
    id2label = dict(zip(range(len(labels)), labels))

    def convert_to_label(ids):
        return list(map(lambda x: id2label[x], ids))

    trainer.init_iterator(set_type)
    predictions = []
    output_label_ids = []
    masks = []
    for _ in range(steps):
        if set_type == 'dev':
            loss, pred, label_id, mask = trainer.eval_step()
        else:
            pred, label_id, mask = trainer.test_step()
        predictions.extend(pred.tolist())
        output_label_ids.extend(label_id.tolist())
        masks.extend(mask.tolist())

    y_pred, y_true = [], []
    for prediction, output_label_id, mask in zip(predictions, output_label_ids, masks):
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
    return y_true, y_pred


def get_model_fn(model_type, config, num_classes, add_crf):
    def model_fn(inputs, is_training):
        model = TokenClassification(
            model_type=model_type,
            config=config,
            num_classes=num_classes,
            is_training=is_training,
            input_ids=inputs['input_ids'],
            input_mask=inputs['input_mask'],
            token_type_ids=inputs['token_type_ids'],
            label_ids=inputs['label_ids'],
            dropout_prob=0.1,
            add_crf=add_crf
        )
        outputs = [model.predictions, inputs['label_ids'], inputs['input_mask']]
        loss = model.loss / gradient_accumulation_steps
        return {'loss': loss, 'outputs': outputs}

    return model_fn


def main():
    set_seed(random_seed)

    config = CONFIGS[model_type].from_pretrained(bert_dir)
    tokenizer = TOKENIZERS[model_type].from_pretrained(bert_dir, do_lower_case=True)

    train_dataset, num_train_batch = load_dataset('train', tokenizer)
    dev_dataset, num_dev_batch = load_dataset('dev', tokenizer)
    test_dataset, num_test_batch = load_dataset('test', tokenizer)

    output_types, output_shapes = return_types_and_shapes(for_trainer=True)

    trainer = Trainer(
        model_type, output_types, output_shapes, device='gpu', use_xla=use_xla, use_torch_mode=use_torch_mode
    )

    trainer.build_model(get_model_fn(model_type, config, len(labels), add_crf))

    t_total = num_train_batch * epochs // gradient_accumulation_steps

    train_op = create_optimizer(
        init_lr=learning_rate,
        gradients=trainer.gradients,
        variables=trainer.variables,
        num_train_steps=t_total,
        num_warmup_steps=t_total * 0.1)

    trainer.compile(
        train_op=train_op, max_checkpoints=1)

    trainer.build_handle(train_dataset, 'train')
    trainer.build_handle(dev_dataset, 'dev')
    trainer.build_handle(test_dataset, 'test')

    trainer.from_pretrained(bert_dir)

    best_score = 0.

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num epochs = {}".format(epochs))
    tf.logging.info("  batch size = {}".format(batch_size))
    tf.logging.info("  Gradient Accumulation steps = {}".format(gradient_accumulation_steps))
    tf.logging.info("  Total train batch size (accumulation) = {}".format(batch_size * gradient_accumulation_steps))
    tf.logging.info("  optimizer steps = %d", t_total)
    tf.logging.info("  Num devices = {}".format(trainer.num_devices))
    tf.logging.info("  Num params = {}".format(trainer.num_params))

    for epoch in range(epochs):
        epoch_iter = bar_fn(range(num_train_batch), desc='epoch {} '.format(epoch + 1))
        for step in epoch_iter:
            if use_torch_mode:
                train_loss = trainer.backward()
            else:
                train_loss = trainer.train_step()
            epoch_iter.set_description(desc='epoch {} ,loss {:.4f}'.format(epoch + 1, train_loss))

            if (step + 1) % gradient_accumulation_steps == 0:
                if use_torch_mode:
                    trainer.train_step()
                    trainer.zero_grad()
                if trainer.global_step % logging_steps == 0 or trainer.global_step == t_total:
                    y_true, y_pred = predict(trainer, num_dev_batch, 'dev')
                    p, r, f = prf_score(y_true, y_pred)
                    if f > best_score:
                        best_score = f
                        trainer.save_pretrained(output_dir)
                        config.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                    tf.logging.info("***** eval results *****")
                    tf.logging.info(" global step : {}".format(trainer.global_step))
                    tf.logging.info(" eval precision score : {:.4f}".format(p))
                    tf.logging.info(" eval recall score : {:.4f}".format(r))
                    tf.logging.info(" eval f1 score : {:.4f}".format(f))
                    tf.logging.info(" best f1 score : {:.4f}".format(best_score))

    tf.logging.info("***** Running Test *****")
    trainer.from_pretrained(output_dir)
    y_true, y_pred = predict(trainer, num_test_batch, 'test')
    report = ner_report(y_true, y_pred)
    tf.logging.info("***** test results *****")
    report = report.split('\n')
    for r in report:
        tf.logging.info(r)


if __name__ == '__main__':
    main()
