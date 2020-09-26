# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: run_ptm.py
@date: 2020/09/21
"""
import platform
import tensorflow.compat.v1 as tf
from textToy.data.classification import (
    InputExample, convert_examples_to_features,
    create_dataset_by_gen, return_types_and_shapes)
from textToy import (SequenceClassification,
                     CONFIGS, TOKENIZERS,
                     set_seed, ProgressBar)
from tqdm import tqdm
import pandas as pd
from textToy.optimizer import create_optimizer
from textToy import Trainer
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import os

if platform.system() == 'Windows':
    bar_fn = ProgressBar  # win10下我使用tqdm老换行，所以自己写了一个
else:
    bar_fn = tqdm  # linux就用tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model_type = 'bert'
data_dir = 'data/classification'
model_dir = 'bert_base'
output_dir = "ckpt/classification"

# batch size 需要是卡数的整倍数
batch_size = 4
gradient_accumulation_steps = 8  # 梯度累积步数
max_seq_length = 32
learning_rate = 2e-5
random_seed = 42
threads = 8  # 数据处理进程数
epochs = 3
labels = ['体育', '娱乐', '家居', '房产', '教育']
logging_steps = 500  # 验证间隔步数，每个多少步验证一次模型

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_examples(filename):
    examples = []
    datas = pd.read_csv(filename, encoding='utf-8', sep='\t').values.tolist()
    for data in datas:
        examples.append(InputExample(
            guid=0, text_a=data[1], label=data[0]
        ))
    return examples


def predict(trainer, steps, set_type='dev'):
    # 预测时，先初始化 dataset，参数传入'dev' or 'test'
    trainer.init_iterator(set_type)
    predictions = None
    output_label_ids = None
    for _ in tqdm(range(steps)):
        if set_type == 'dev':
            # 验证会返回loss
            loss, pred, label_id = trainer.eval_step()
        else:
            # 预测不会返回loss
            pred, label_id = trainer.test_step()
        if predictions is None:
            predictions = pred
            output_label_ids = label_id
        else:
            predictions = np.append(predictions, pred, 0)
            output_label_ids = np.append(output_label_ids, label_id, 0)
    predictions = np.argmax(predictions, -1)
    return output_label_ids, predictions


def load_dataset(set_type, tokenizer):
    examples = create_examples(os.path.join(data_dir, set_type + '.csv'))
    features = convert_examples_to_features(examples, tokenizer,
                                            max_length=max_seq_length, set_type=set_type,
                                            label_list=labels, use_multi_threads=True, threads=threads)
    dataset, steps_one_epoch = create_dataset_by_gen(features, batch_size, set_type)
    return dataset, steps_one_epoch


def get_model_fn(model_type, config, num_classes):
    def model_fn(inputs, is_training):
        model = SequenceClassification(
            model_type=model_type, config=config,
            num_classes=num_classes, is_training=is_training,
            **inputs)
        outputs = [model.logits, inputs['label_ids']]
        loss = model.loss / gradient_accumulation_steps
        return {'loss': loss, 'outputs': outputs}

    return model_fn


def main():
    set_seed(random_seed)
    config = CONFIGS[model_type].from_pretrained(model_dir)
    tokenizer = TOKENIZERS[model_type].from_pretrained(model_dir, do_lower_case=True)

    # 创建dataset
    train_dataset, num_train_batch = load_dataset('train', tokenizer)
    dev_dataset, num_dev_batch = load_dataset('dev', tokenizer)
    test_dataset, num_test_batch = load_dataset('test', tokenizer)

    output_types, output_shapes = return_types_and_shapes(for_trainer=True)

    # 初始化trainer
    trainer = Trainer(
        model_type, output_types, output_shapes, device='gpu'
    )

    # 构建模型
    trainer.build_model(get_model_fn(model_type, config, num_classes=len(labels)))

    t_total = num_train_batch * epochs // gradient_accumulation_steps
    # 创建train op
    train_op = create_optimizer(init_lr=learning_rate,
                                gradients=trainer.gradients,
                                variables=trainer.variables,
                                num_train_steps=t_total,
                                num_warmup_steps=int(t_total * epochs * 0.1))

    # 配置trainer，将train op、模型保存最大数量传入
    trainer.compile(train_op, max_checkpoints=1)

    trainer.build_handle(train_dataset, 'train')
    trainer.build_handle(dev_dataset, 'dev')
    trainer.build_handle(test_dataset, 'test')

    # 预训练模型加载预训练参数，若不加载，调用trainer.init_variables()
    trainer.from_pretrained(model_dir)

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
            train_loss = trainer.backward()
            epoch_iter.set_description(desc='epoch {} ,loss {:.4f}'.format(epoch + 1, train_loss))

            if (step + 1) % gradient_accumulation_steps == 0:
                trainer.train_step()
                trainer.zero_grad()

                if trainer.global_step % logging_steps == 0 or trainer.global_step == t_total:
                    y_true, y_pred = predict(trainer, num_dev_batch, 'dev')
                    acc = accuracy_score(y_true, y_pred)
                    if acc > best_score:
                        best_score = acc
                        trainer.save_pretrained(output_dir)
                        config.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                    tf.logging.info("***** eval results *****")
                    tf.logging.info(" global step : {}".format(trainer.global_step))
                    tf.logging.info(" eval accuracy : {:.4f}".format(acc))
                    tf.logging.info(" best accuracy : {:.4f}".format(best_score))

    tf.logging.info("***** Running Test *****")
    trainer.from_pretrained(output_dir)
    y_true, y_pred = predict(trainer, num_test_batch, 'test')
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    tf.logging.info("***** test results *****")
    report = report.split('\n')
    for r in report:
        tf.logging.info(r)


if __name__ == '__main__':
    main()