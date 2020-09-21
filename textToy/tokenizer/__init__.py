# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: __init__.py.py
@date: 2020/09/08
"""
from .base import (
    PTMTokenizer, BasicTokenizer, WordpieceTokenizer)
from .bert import BertTokenizer
from .albert import ALBertTokenizer
from .bert import BertTokenizer as NeZhaTokenizer
from .bert import BertTokenizer as ElectraTokenizer
from .wobert import WoBertTokenizer


TOKENIZERS = {
    'bert': BertTokenizer, 'albert': ALBertTokenizer,
    'nezha': BertTokenizer, 'electra': BertTokenizer,
    'wobert': WoBertTokenizer
}
