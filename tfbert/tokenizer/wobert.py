# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: wobert.py
@date: 2020/09/19
"""
from .tokenization_base import convert_to_unicode
from .bert import BertTokenizer


class WoBertTokenizer(BertTokenizer):
    def __init__(self, seg_fn=None, **kwargs):
        super(WoBertTokenizer, self).__init__(**kwargs)
        import jieba
        if seg_fn is None:
            self.seg_fn = lambda x: jieba.cut(x, HMM=False)
        else:
            self.seg_fn = seg_fn

    def tokenize(self, text):
        text = convert_to_unicode(text)
        split_tokens = []
        for token in self.seg_fn(text):
            if token in self.vocab:
                split_tokens.append(token)
            else:
                for t in self.basic_tokenizer.tokenize(token):
                    for sub_token in self.wordpiece_tokenizer.tokenize(t):
                        split_tokens.append(sub_token)
        return split_tokens
