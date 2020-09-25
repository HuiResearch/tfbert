# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: bert.py
@date: 2020/09/08
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from . import PTMTokenizer
from .base import load_vocab, BasicTokenizer, WordpieceTokenizer
import tensorflow.compat.v1 as tf


class BertTokenizer(PTMTokenizer):
    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.do_lower_case = do_lower_case
        self.num_special_tokens = 2

    @property
    def vocab_size(self):
        return len(self.vocab)

    def convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    @classmethod
    def from_pretrained(cls, vocab_dir_or_file, **kwargs):
        do_lower_case = kwargs.pop('do_lower_case', True)
        if os.path.isdir(vocab_dir_or_file):
            filename = 'vocab.txt'
            vocab_file = os.path.join(vocab_dir_or_file, filename)
        else:
            vocab_file = vocab_dir_or_file

        return cls(vocab_file=vocab_file, do_lower_case=do_lower_case, **kwargs)

    def save_pretrained(self, save_directory):
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, 'vocab.txt')
        else:
            vocab_file = save_directory

        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for token, index in self.vocab.items():
                writer.write(token.strip() + '\n')
        tf.logging.info("  Tokenizer vocab saved in {}".format(vocab_file))
        return vocab_file

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
            # If the token is part of the never_split set
            if token in self.basic_tokenizer.never_split:
                split_tokens.append(token)
            else:
                split_tokens += self.wordpiece_tokenizer.tokenize(token)

        return split_tokens
