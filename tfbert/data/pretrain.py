# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: pretrain.py
@date: 2021/03/22
"""
from . import BaseClass, multiple_convert_examples_to_features, single_example_to_features
from ..tokenizer.tokenization_base import convert_to_unicode
from ..tokenizer import GlyceBertTokenizer
import random
import numpy as np
import collections
from functools import partial

CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"


class InputExample(BaseClass):
    def __init__(self,
                 tokens,
                 token_type_ids,
                 masked_lm_positions=None,
                 masked_lm_labels=None,
                 is_random_next=None):
        self.tokens = tokens
        self.token_type_ids = token_type_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.is_random_next = is_random_next


class InputFeature(BaseClass):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 attention_mask=None,
                 token_type_ids=None,
                 pinyin_ids=None,
                 masked_lm_positions=None,
                 masked_lm_ids=None,
                 masked_lm_weights=None,
                 next_sentence_labels=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.pinyin_ids = pinyin_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids
        self.masked_lm_weights = masked_lm_weights
        self.next_sentence_labels = next_sentence_labels


def convert_example_to_feature_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_example_to_feature(
        example: InputExample, max_length, max_predictions_per_seq, only_mlm=False):
    input_ids = tokenizer.convert_tokens_to_ids(example.tokens)
    attention_mask = [1] * len(input_ids)
    token_type_ids = list(example.token_type_ids)

    assert len(input_ids) <= max_length

    while len(input_ids) < max_length:
        input_ids.append(0)
        attention_mask.append(0)
        token_type_ids.append(0)
    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(token_type_ids) == max_length

    masked_lm_positions = list(example.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(example.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)
    if not only_mlm:
        sentence_order_label = 1 if example.is_random_next else 0
    else:
        sentence_order_label = None

    if isinstance(tokenizer, GlyceBertTokenizer):
        pinyin_ids = tokenizer.convert_token_ids_to_pinyin_ids(input_ids)
    else:
        pinyin_ids = None
    feature = InputFeature(
        guid=0,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        pinyin_ids=pinyin_ids,
        masked_lm_positions=masked_lm_positions,
        masked_lm_ids=masked_lm_ids,
        masked_lm_weights=masked_lm_weights,
        next_sentence_labels=sentence_order_label
    )
    return feature


def convert_examples_to_features(
        examples, tokenizer, max_length, max_predictions_per_seq, only_mlm=False, set_type='train', threads=1):
    annotate_ = partial(
        convert_example_to_feature,
        max_length=max_length,
        max_predictions_per_seq=max_predictions_per_seq,
        only_mlm=only_mlm
    )
    if threads > 1:
        features = multiple_convert_examples_to_features(
            examples,
            annotate_=annotate_,
            initializer=convert_example_to_feature_init,
            initargs=(tokenizer,),
            threads=threads
        )
    else:
        convert_example_to_feature_init(tokenizer)
        features = single_example_to_features(
            examples, annotate_=annotate_
        )
    new_features = []
    i = 0
    for feature in features:
        feature.guid = set_type + '-' + str(i)
        new_features.append(feature)
    return new_features


def create_examples(
        input_files, tokenizer, max_length,
        dupe_factor, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq,
        random_next_sentence=False,
        do_whole_word_mask=False,
        favor_shorter_ngram=True,
        ngram=3):
    all_documents = [[]]
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = convert_to_unicode(line)
                if not line:
                    break
                line = line.strip()

                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)
    all_documents = [x for x in all_documents if x]
    random.shuffle(all_documents)
    vocab_words = list(tokenizer.vocab.keys())
    examples = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            examples.extend(create_examples_from_document(
                all_documents, document_index, max_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                vocab_words, random_next_sentence, do_whole_word_mask, favor_shorter_ngram, ngram
            ))
    random.shuffle(examples)
    return examples


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_examples_from_document(
        all_documents, document_index, max_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq,
        vocab_words, random_next_sentence=False,
        do_whole_word_mask=False,
        favor_shorter_ngram=True, ngram=3
):
    document = all_documents[document_index]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    examples = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or \
                        (random_next_sentence and random.random() < 0.5):
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = random.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break

                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                elif not random_next_sentence and random.random() < 0.5:
                    is_random_next = True
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
                    tokens_a, tokens_b = tokens_b, tokens_a
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                token_type_ids = []
                tokens.append(CLS_TOKEN)
                token_type_ids.append(0)

                for token in tokens_a:
                    tokens.append(token)
                    token_type_ids.append(0)
                tokens.append(SEP_TOKEN)
                token_type_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    token_type_ids.append(1)
                tokens.append(SEP_TOKEN)
                token_type_ids.append(1)
                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words,
                    do_whole_word_mask,
                    favor_shorter_ngram, ngram
                )
                example = InputExample(
                    tokens=tokens,
                    token_type_ids=token_type_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels
                )
                examples.append(example)
            current_chunk = []
            current_length = 0
        i += 1
    return examples


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(
        tokens, masked_lm_prob,
        max_predictions_per_seq,
        vocab_words, do_whole_word_mask=False,
        favor_shorter_ngram=True, ngram=3):
    cand_indexes = []

    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    for (i, token) in enumerate(tokens):
        if token == CLS_TOKEN or token == SEP_TOKEN:
            continue

        if do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    output_tokens = list(tokens)
    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return output_tokens, masked_lm_positions, masked_lm_labels

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    if not favor_shorter_ngram:
        pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    random.shuffle(ngram_indexes)
    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(ngrams[:len(cand_index_set)],
                             p=pvals[:len(cand_index_set)] /
                               pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = MASK_TOKEN
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    random.shuffle(ngram_indexes)
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return output_tokens, masked_lm_positions, masked_lm_labels
