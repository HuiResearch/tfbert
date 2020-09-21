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
from . import PTMTokenizer
from .base import (
    load_vocab, BasicTokenizer, WordpieceTokenizer, printable_text)
import os
from shutil import copyfile

SPIECE_UNDERLINE = u"▁".encode("utf-8")


class ALBertTokenizer(PTMTokenizer):
    spm_model_filenames = []

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
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)

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

