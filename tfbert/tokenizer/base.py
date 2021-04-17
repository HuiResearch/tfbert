# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: base.py
@date: 2020/09/08
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata
import collections
import six
import tensorflow.compat.v1 as tf


class PTMTokenizer(object):
    SPECIAL_TOKENS_ATTRIBUTES = [
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token"
    ]

    @property
    def unk_token(self):
        """ Unknown token (string). Log an error if used while not having been set. """
        if self._unk_token is None:
            tf.logging.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def sep_token(self):
        """ Separation token (string). E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
        if self._sep_token is None:
            tf.logging.error("Using sep_token, but it is not set yet.")
        return self._sep_token

    @property
    def pad_token(self):
        """ Padding token (string). Log an error if used while not having been set. """
        if self._pad_token is None:
            tf.logging.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def cls_token(self):
        """ Classification token (string). E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the ptm. Log an error if used while not having been set. """
        if self._cls_token is None:
            tf.logging.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @property
    def mask_token(self):
        """ Mask token (string). E.g. when training a ptm with masked-language modeling. Log an error if used while not having been set. """
        if self._mask_token is None:
            tf.logging.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @property
    def unk_token_id(self):
        """ Id of the unknown token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self):
        """ Id of the separation token in the vocabulary. E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        """ Id of the padding token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self):
        """ Id of the padding token type in the vocabulary."""
        return self._pad_token_type_id

    @property
    def cls_token_id(self):
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the ptm. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self):
        """ Id of the mask token in the vocabulary. E.g. when training a ptm with masked-language modeling. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def special_tokens_map_extended(self):
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            Convert tokens of AddedToken type in string.
            All returned tokens are strings
            (cls_token, unk_token...).
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            Keep the tokens as AddedToken if they are of this type.

            AddedToken can be used to control more finely how special tokens are tokenized.
        """
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    def __init__(self, **kwargs):

        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0

        self.num_special_tokens = 2

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                assert isinstance(value, str)
                setattr(self, key, value)

    def convert_tokens_to_ids(self, tokens):
        """ Converts a single token, or a sequence of tokens, (str) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self.convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self.convert_token_to_id(token))
        return ids

    def convert_token_to_id(self, token):
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids):
        if ids is None:
            return None

        if isinstance(ids, int):
            return self.convert_id_to_token(ids)

        tokens = []
        for id_ in ids:
            tokens.append(self.convert_id_to_token(id_))
        return tokens

    def convert_id_to_token(self, id):
        raise NotImplementedError

    def tokenize(self, text):
        raise NotImplementedError

    def truncate(self,
                 first,
                 max_length,
                 second=None,
                 add_special_tokens=True
                 ):
        if add_special_tokens:
            if second:
                max_length = max_length - self.num_special_tokens - 1  # 双句任务去掉 [CLS] [SEP] [SEP]
            else:
                max_length = max_length - self.num_special_tokens  # 单句任务去掉 [CLS] [SEP]
        if second:
            while True:
                total_length = len(first) + len(second)
                if total_length <= max_length:
                    break
                if len(first) > len(second):
                    first.pop()
                else:
                    second.pop()
            return first, second
        else:
            if len(first) <= max_length:
                first = first
            else:
                first = first[: max_length]
            return first, None

    def get_input_ids(self, text):
        tokens = self.tokenize(text)
        input_ids = self.convert_tokens_to_ids(tokens)
        return input_ids

    def __call__(self,
                 text,
                 text_pair=None,
                 add_special_tokens=True,
                 max_length=None,
                 pad_to_max_length=False):
        return self.encode(
            text,
            text_pair,
            add_special_tokens,
            max_length,
            pad_to_max_length
        )

    def encode(self,
               text,
               text_pair=None,
               add_special_tokens=True,
               max_length=None,
               pad_to_max_length=False):
        '''
        将文本转为bert输入，返回是一个字典，{'input_ids': , 'input_mask': , 'token_type_ids': }
        :param text: 第一句
        :param text_pair: 第二句
        :param add_special_tokens: 是否加上cls sep
        :param max_length: 最大长度，若是设置了最大长度，就会进行裁剪
        :param pad_to_max_length:  是否将结果pad到最大长度
        :return:
        '''

        first_ids = self.get_input_ids(text)
        second_ids = self.get_input_ids(text_pair) if text_pair is not None else None
        return self.prepare_inputs(
            ids=first_ids,
            pair_ids=second_ids,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            pad_to_max_length=pad_to_max_length,
        )

    def prepare_inputs(self,
                       ids,
                       pair_ids=None,
                       max_length=None,
                       add_special_tokens=True,
                       pad_to_max_length=False,
                       ):

        encoded_inputs = {}

        if max_length:
            ids, pair_ids = self.truncate(ids, max_length=max_length, second=pair_ids,
                                          add_special_tokens=add_special_tokens)
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
            if pair_ids:
                pair_ids = pair_ids + [self.sep_token_id]
        input_ids = ids + pair_ids if pair_ids else ids

        token_type_ids = [0] * len(ids)
        if pair_ids:
            token_type_ids = token_type_ids + [1] * len(pair_ids)
        input_mask = [1] * len(input_ids)

        if pad_to_max_length:
            while len(input_ids) < max_length:
                input_ids.append(self.pad_token_id)
                input_mask.append(0)
                token_type_ids.append(self.pad_token_type_id)
        encoded_inputs['input_ids'] = input_ids
        encoded_inputs['input_mask'] = input_mask
        encoded_inputs['token_type_ids'] = token_type_ids

        return encoded_inputs


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True):
        """ Constructs a BasicTokenizer.

        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text, never_split=None):
        """ Basic Tokenization of a piece of text.
            Split on "white spaces" only, for sub-word tokenization, see WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return six.ensure_text(text, "utf-8", "ignore")
        elif isinstance(text, six.text_type):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, six.text_type):
            return six.ensure_binary(text, "utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='utf-8') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens
