# -*- coding:utf-8 -*-
# @FileName  :glyce_bert.py
# @Time      :2021/7/29 18:19
# @Author    :huanghui
import os
import tensorflow.compat.v1 as tf
import json
from .tokenization_base import convert_to_unicode, PaddingStrategy, TruncationStrategy
from .bert import BertTokenizer
from typing import List, Union, Tuple, Optional


class GlyceBertTokenizer(BertTokenizer):
    def __init__(self, config_path, **kwargs):
        super(GlyceBertTokenizer, self).__init__(**kwargs)
        # load pinyin map dict
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def save_pretrained(self, save_directory):

        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, 'vocab.txt')
            config_path = os.path.join(save_directory, 'config')
        else:
            vocab_file = save_directory
            config_path = os.path.join(os.path.split(save_directory)[0], "config")

        if not os.path.exists(config_path):
            os.makedirs(config_path)

        with open(os.path.join(config_path, 'pinyin_map.json'), "w", encoding='utf8') as fin:
            fin.write(json.dumps(self.pinyin_dict, ensure_ascii=False))

        with open(os.path.join(config_path, 'id2pinyin.json'), "w", encoding='utf8') as fin:
            fin.write(json.dumps(self.id2pinyin, ensure_ascii=False))

        with open(os.path.join(config_path, 'pinyin2tensor.json'), "w", encoding='utf8') as fin:
            fin.write(json.dumps(self.pinyin2tensor, ensure_ascii=False))

        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for token, index in self.vocab.items():
                writer.write(token.strip() + '\n')
        tf.logging.info("  Tokenizer vocab saved in {}".format(vocab_file))
        return vocab_file

    @classmethod
    def from_pretrained(cls, vocab_dir_or_file, **kwargs):
        do_lower_case = kwargs.pop('do_lower_case', True)
        if os.path.isdir(vocab_dir_or_file):
            filename = 'vocab.txt'
            vocab_file = os.path.join(vocab_dir_or_file, filename)
            config_path = os.path.join(vocab_dir_or_file, "config")
        else:
            vocab_file = vocab_dir_or_file
            config_path = os.path.join(os.path.split(vocab_dir_or_file)[0], "config")

        return cls(config_path=config_path, vocab_file=vocab_file, do_lower_case=do_lower_case, **kwargs)

    def convert_token_ids_to_pinyin_ids(self, ids):
        from pypinyin import pinyin, Style

        tokens = self.convert_ids_to_tokens(ids)
        offsets = []
        pos = 0
        sentence = ""
        for token in tokens:
            token = token.replace("##", "").strip()

            if len(token) == 0:
                token = " "
            if token in self.all_special_tokens:
                token = " "
                offsets.append((0, 0))
            else:
                offsets.append((pos, pos + len(token)))
            pos += len(token)
            sentence += token

        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, offset in enumerate(offsets):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids

    def _encode_plus(
            self,
            text: Union[str, List[str], List[int]],
            text_pair: Optional[Union[str, List[str], List[int]]] = None,
            add_special_tokens: bool = True,
            padding_strategy: Union[bool, str, PaddingStrategy] = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: Union[bool, str, TruncationStrategy] = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_length: bool = False,
    ):
        first_ids = self.get_input_ids(text)
        second_ids = self.get_input_ids(text_pair) if text_pair is not None else None
        encoded = self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy,
            truncation=truncation_strategy,
            max_length=max_length,
            stride=stride,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length
        )
        pinyin_ids = self.convert_token_ids_to_pinyin_ids(encoded['input_ids'])
        assert len(pinyin_ids) == len(encoded['input_ids'])
        encoded['pinyin_ids'] = pinyin_ids
        return encoded

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[str],
                List[Tuple[str, str]],
                List[Tuple[List[str], List[str]]],
                List[Tuple[str, str]],
                List[List[int]],
                List[Tuple[List[int], List[int]]],
            ],
            add_special_tokens: bool = True,
            padding_strategy: Union[bool, str, PaddingStrategy] = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: Union[bool, str, TruncationStrategy] = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_length: bool = False,
    ):
        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = self.get_input_ids(ids)
            second_ids = self.get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length
        )
        batch_pinyin_ids = []
        for i in batch_outputs['input_ids']:
            pinyin_ids = self.convert_token_ids_to_pinyin_ids(batch_outputs['input_ids'][i])
            assert len(pinyin_ids) == len(batch_outputs['input_ids'][i])
            batch_pinyin_ids.append(pinyin_ids)
        batch_outputs['pinyin_ids'] = batch_pinyin_ids
        return batch_outputs
