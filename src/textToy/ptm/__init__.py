# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: __init__.py.py
@date: 2020/09/08
"""
from .bert import BertModel
from .bert import BertModel as WoBertModel
from .albert import ALBertModel
from .nezha import NeZhaModel
from .electra import ELECTRAModel

MODELS = {
    'bert': BertModel, 'albert': ALBertModel,
    'nezha': NeZhaModel, 'electra': ELECTRAModel,
    'wobert': WoBertModel
}
from .finetune import (
    SequenceClassification, TokenClassification,
    PTMExtractFeature, MultiLabelClassification, PretrainingModel)
