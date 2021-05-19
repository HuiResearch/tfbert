# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: albert.py
@date: 2020/09/08
"""
import six
import collections
import unicodedata
import tensorflow.compat.v1 as tf
from .tokenization_base import PTMTokenizer
from .tokenization_base import (
    load_vocab, BasicTokenizer, WordpieceTokenizer, printable_text)
import os
from shutil import copyfile

SPIECE_UNDERLINE = u"▁".encode("utf-8")


class ALBertTokenizer(PTMTokenizer):
    spm_model_filenames = []
    padding_side = 'right'
    model_max_length = 512
    model_input_names = ['input_ids', 'token_type_ids', 'attention_mask']

    def __init__(self,
                 vocab_file=None,
                 do_lower_case=True,
                 spm_model_file=None,
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
            **kwargs)

        if vocab_file is None and spm_model_file is None:
            raise ValueError("At least one of vocab_file and spm_model_file must be set")

        self.vocab_file = vocab_file
        self.spm_model_file = spm_model_file

        self.vocab = None
        self.sp_model = None
        self.do_lower_case = do_lower_case
        if spm_model_file:
            import sentencepiece as spm
            self.sp_model = spm.SentencePieceProcessor()

            # Handle cases where SP can't load the file, but gfile can.
            sp_model_ = tf.gfile.GFile(spm_model_file, "rb").read()
            self.sp_model.LoadFromSerializedProto(sp_model_)
            # Note(mingdachen): For the purpose of consisent API, we are
            # generating a vocabulary for the sentence piece tokenizer.
            self.vocab = {self.sp_model.IdToPiece(i): i for i
                          in range(self.sp_model.GetPieceSize())}
        else:
            self.vocab = load_vocab(vocab_file)
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.num_special_tokens = 2

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return self.vocab_size

    def convert_token_to_id(self, token):
        if self.sp_model:
            return self.sp_model.PieceToId(
                printable_text(token))
        else:
            return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_id_to_token(self, index):
        if self.sp_model:
            return self.sp_model.IdToPiece(index)
        else:
            return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        if self.sp_model:
            out_string = "".join(tokens).replace('▁', " ").strip()
        else:
            out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.
        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.
        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.
        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
            if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        A BERT sequence has the following format:
        ::
            - single sequence: ``[CLS] X [SEP]``
            - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        A BERT offset_mapping has the following format:
        ::
            - single sequence: ``(0,0) X (0,0)``
            - pair of sequences: `(0,0) A (0,0) B (0,0)``

        Args:
            offset_mapping_0 (:obj:`List[tuple]`):
                List of char offsets to which the special tokens will be added.
            offset_mapping_1 (:obj:`List[tuple]`, `optional`):
                Optional second list of char offsets for offset mapping pairs.
        Returns:
            :obj:`List[tuple]`: List of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                          _sep) * [1]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.
        Args:
            token_ids_0 (List[int]): List of ids of the first sequence.
            token_ids_1 (List[int], optinal): List of ids of the second sequence.
            already_has_special_tokens (bool, optional): Whether or not the token list is already
                formatted with special tokens for the model. Defaults to None.
        Returns:
            results (List[int]): The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                    [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    @classmethod
    def from_pretrained(cls, vocab_dir_or_file=None, **kwargs):
        do_lower_case = kwargs.pop('do_lower_case', True)
        spm_model_file = None
        vocab_file = None

        if os.path.isdir(vocab_dir_or_file):
            filenames = os.listdir(vocab_dir_or_file)
            for filename in filenames:
                if ".ptm" in filename:
                    spm_model_file = os.path.join(vocab_dir_or_file, filename)
                    break

                if 'vocab.txt' in filename:
                    vocab_file = os.path.join(vocab_dir_or_file, filename)
        else:
            if 'vocab.txt' in vocab_dir_or_file:
                vocab_file = vocab_dir_or_file
            else:
                spm_model_file = vocab_dir_or_file

        return cls(vocab_file=vocab_file, spm_model_file=spm_model_file,
                   do_lower_case=do_lower_case, **kwargs)

    def save_vocab(self, save_directory):
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, 'vocab.txt')
        else:
            vocab_file = save_directory

        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for token, index in self.vocab.items():
                writer.write(token.strip() + '\n')
        tf.logging.info("Tokenizer vocab saved in {}".format(vocab_file))
        return vocab_file

    def save_spm_model(self, save_directory):

        if os.path.isdir(save_directory):
            out_file = os.path.join(save_directory, 'spiece.ptm')
        else:
            out_file = save_directory

        if os.path.abspath(self.spm_model_file) != os.path.abspath(out_file):
            copyfile(self.spm_model_file, out_file)
        tf.logging.info("Tokenizer spm ptm saved in {}".format(out_file))
        return out_file

    def save_pretrained(self, save_directory):
        if self.sp_model:
            self.save_spm_model(save_directory)
        else:
            self.save_vocab(save_directory)

    def tokenize(self, text):
        if self.sp_model:
            split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
        else:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)

        return split_tokens


def preprocess_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    if six.PY2 and isinstance(outputs, str):
        try:
            outputs = six.ensure_text(outputs, "utf-8")
        except UnicodeDecodeError:
            outputs = six.ensure_text(outputs, "latin-1")

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    """turn sentences into word pieces."""

    if six.PY2 and isinstance(text, six.text_type):
        text = six.ensure_binary(text, "utf-8")

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        piece = printable_text(piece)
        if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = six.ensure_text(piece, "utf-8")
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens
