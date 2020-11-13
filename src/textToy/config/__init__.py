# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: __init__.py.py
@date: 2020/09/08
"""

from .base import BaseConfig
from .ptm import (
    BertConfig, ALBertConfig, ElectraConfig)
from .ptm import BertConfig as NeZhaConfig
from .ptm import BertConfig as WoBertConfig

CONFIGS = {
    'bert': BertConfig, 'albert': ALBertConfig,
    'nezha': BertConfig, 'electra': ElectraConfig,
    'wobert': WoBertConfig
}
