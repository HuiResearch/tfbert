# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2021/1/31 15:16
# @Author    :huanghui

import tensorflow.compat.v1 as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.disable_v2_behavior()

from .models import (
    BertModel, ALBertModel, ElectraModel,
    NezhaModel, WoBertModel, GlyceBertModel,
    SequenceClassification, MODELS, crf,
    TokenClassification, MultiLabelClassification,
    MaskedLM, PretrainingLM, QuestionAnswering)
from .config import (
    BaseConfig, BertConfig, ALBertConfig,
    ElectraConfig, NeZhaConfig, WoBertConfig, GlyceBertConfig, CONFIGS)
from .tokenizer import (
    BasicTokenizer, BertTokenizer, WoBertTokenizer,
    ALBertTokenizer, ElectraTokenizer, NeZhaTokenizer,
    GlyceBertTokenizer, TOKENIZERS)

from .utils import (
    devices, init_checkpoints,
    get_assignment_map_from_checkpoint, ProgressBar,
    clean_bert_model,
    set_seed)

from .optimization import (
    AdamWeightDecayOptimizer, LAMBOptimizer,
    lr_schedule, create_optimizer, create_train_op)
from .data import (Dataset, collate_batch, sequence_padding,
                   single_example_to_features,
                   multiple_convert_examples_to_features,
                   compute_types, compute_shapes,
                   process_dataset, compute_types_and_shapes_from_dataset)
from .trainer import Trainer, SimplerTrainer
