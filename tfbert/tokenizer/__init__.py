# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2021/1/31 15:17
# @Author    :huanghui
from .tokenization_base import (
    PTMTokenizer, BasicTokenizer, WordpieceTokenizer)
from .bert import BertTokenizer
from .albert import ALBertTokenizer
from .bert import BertTokenizer as NeZhaTokenizer
from .bert import BertTokenizer as ElectraTokenizer
from .wobert import WoBertTokenizer

TOKENIZERS = {
    'bert': BertTokenizer, 'albert': ALBertTokenizer,
    'nezha': NeZhaTokenizer, 'electra': ElectraTokenizer,
    'wobert': WoBertTokenizer
}
