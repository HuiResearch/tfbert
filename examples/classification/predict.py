# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: predict.py
@date: 2020/09/21
"""
from textToy import Trainer, BertConfig, SequenceClassification, BertTokenizer
from textToy.data.classification import return_types_and_shapes, convert_examples_to_features, create_dataset_by_gen
from .run_ptm import labels, predict, create_examples, classification_report

output_dir = 'ckpt/classification'
model_type = 'bert'
predict_file = "data/classification/test.csv"
batch_size = 32
max_seq_length = 32

config = BertConfig.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)

output_types, output_shapes = return_types_and_shapes(for_trainer=True)

trainer = Trainer(model_type, output_types, output_shapes)

test_examples = create_examples(predict_file)
test_features = convert_examples_to_features(test_examples, tokenizer, max_seq_length, labels, 'test')
test_dataset, test_steps = create_dataset_by_gen(test_features, batch_size, 'test')

model = SequenceClassification(model_type=model_type,
                               config=config,
                               num_classes=len(labels),
                               is_training=trainer.is_training,
                               **trainer.inputs)

trainer.compile(outputs=[model.logits, trainer.inputs['label_ids']])
trainer.build_handle(test_dataset, 'test')
trainer.from_pretrained(output_dir)

y_true, y_pred = predict(trainer, test_steps, 'test')

report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print(report)