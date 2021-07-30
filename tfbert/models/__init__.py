# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2021/1/31 15:20
# @Author    :huanghui

from .bert import BertModel
from .bert import BertModel as WoBertModel
from .albert import ALBertModel
from .electra import ElectraModel
from .nezha import NezhaModel
from .glyce_bert import GlyceBertModel
from .model_utils import (
    dropout, layer_norm_and_dropout, layer_norm,
    create_weight, get_shape_list, gather_indexes, create_initializer)

from .embeddings import (create_word_embeddings, create_position_embeddings, create_token_type_embeddings)
from . import crf


MODELS = {
    'bert': BertModel,
    'albert': ALBertModel,
    'electra': ElectraModel,
    'wobert': WoBertModel,
    'nezha': NezhaModel,
    'glyce_bert': GlyceBertModel
}

from .for_task import (
    SequenceClassification, TokenClassification,
    MultiLabelClassification, MaskedLM, PretrainingLM,
    QuestionAnswering)

