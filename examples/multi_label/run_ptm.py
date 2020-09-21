# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: run_ptm.py
@date: 2020/09/21
"""
import os
import platform
import tensorflow.compat.v1 as tf
from textToy.data.classification import (
    InputExample, convert_examples_to_features,
    create_dataset_by_gen, return_types_and_shapes)
from textToy import (MultiDeviceTrainer, MultiLabelClassification,
                     CONFIGS, TOKENIZERS,
                     set_seed, ProgressBar)
from tqdm import tqdm
import json
from textToy.ptm.ckpt_utils import get_save_vars
from textToy.optimizer import create_optimizer
import numpy as np
from textToy.metric.multi_label import multi_label_metric

if platform.system() == 'Windows':
    bar_fn = ProgressBar
else:
    bar_fn = tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model_type = 'bert'
data_dir = 'data/multi_label'
model_dir = 'bert_base'
output_dir = "ckpt/multi_label"

batch_size = 32
max_seq_length = 128
learning_rate = 2e-5
random_seed = 42
threads = 8
epochs = 4
labels = ['DV%d' % (i + 1) for i in range(20)]
threshold = [0.5] * len(labels)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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
    examples = create_examples(os.path.join(data_dir, set_type + '.json'))

    features = convert_examples_to_features(examples, tokenizer,
                                            max_length=max_seq_length, set_type=set_type,
                                            label_list=labels, is_multi_label=True,
                                            use_multi_threads=True, threads=threads)
    dataset, steps_one_epoch = create_dataset_by_gen(features, batch_size, set_type, is_multi_label=True)
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
        loss = model.loss
        return {'loss': loss, 'outputs': outputs}

    return model_fn


def main():
    set_seed(random_seed)
    config = CONFIGS[model_type].from_pretrained(model_dir)
    tokenizer = TOKENIZERS[model_type].from_pretrained(model_dir, do_lower_case=True)

    train_dataset, train_steps = load_dataset('train', tokenizer)
    test_dataset, test_steps = load_dataset('test', tokenizer)

    output_types, output_shapes = return_types_and_shapes(for_trainer=True, is_multi_label=True)

    trainer = MultiDeviceTrainer(
        model_type, output_types, output_shapes, device='gpu'
    )

    trainer.build_model(get_model_fn(model_type, config, len(labels)))

    train_op = create_optimizer(
        trainer.loss, init_lr=learning_rate,
        num_train_steps=train_steps * epochs,
        num_warmup_steps=train_steps * epochs * 0.1,
        grads_and_vars=trainer.grads_and_vars)

    trainer.compile(
        train_op, var_list=get_save_vars(), max_checkpoints=1)

    trainer.build_handle(train_dataset, 'train')
    trainer.build_handle(test_dataset, 'test')

    trainer.from_pretrained(model_dir)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  batch size = {}".format(batch_size))
    tf.logging.info("  epochs = {}".format(epochs))
    tf.logging.info("  optimizer steps = %d", train_steps * epochs)
    tf.logging.info("  num devices = {}".format(trainer.num_devices))
    tf.logging.info("  num params = {}".format(trainer.num_params))

    best_score = 0.
    for epoch in range(epochs):
        epoch_iter = bar_fn(range(train_steps), desc='epoch {} '.format(epoch + 1))
        for step in epoch_iter:
            train_loss = trainer.train_step()
            epoch_iter.set_description(desc='epoch {} ,loss {:.4f}'.format(epoch + 1, train_loss))
            # if trainer.global_step % logging_steps == 0 or trainer.global_step == train_steps * epochs:
        y_true, y_pred = predict(trainer, test_steps, 'test')
        score = multi_label_metric(y_true, y_pred, label_list=labels)['dict_result']['micro macro avg']['f1-score']
        if score > best_score:
            best_score = score
            trainer.save_pretrained(output_dir)
            config.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        tf.logging.info("***** eval results *****")
        tf.logging.info(" global step : {}".format(trainer.global_step))
        tf.logging.info(" eval score : {:.4f}".format(score))
        tf.logging.info(" best score : {:.4f}".format(best_score))

    tf.logging.info("***** Running Test *****")
    trainer.from_pretrained(output_dir)
    y_true, y_pred = predict(trainer, test_steps, 'test')
    report = multi_label_metric(y_true, y_pred, label_list=labels)['string_result']
    open(os.path.join(output_dir, 'result.txt'), 'w', encoding='utf-8').write(report)
    tf.logging.info("***** test results *****")
    report = report.split('\n')
    for r in report:
        tf.logging.info(r)


if __name__ == '__main__':
    main()
