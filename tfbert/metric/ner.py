# -*- coding: utf-8 -*-
"""
Project Name:   my_project
File Name:      ner
date:           2020/9/13
author:         HuangHui
"""
import numpy as np
from collections import defaultdict


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

        Args:
            seq (list): sequence of labels.

        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).

        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entities(seq)
            [('PER', 0, 1), ('LOC', 3, 3)]
        """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def prf_score(y_true, y_pred, suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
    r = 100 * nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, score


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def ner_metric(y_true, y_pred, digits=4, dict_report=False):
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.
        dict_report :
    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.

    Examples:
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(ner_metric(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          micro avg       0.50      0.50      0.50         2
       weighted avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_longest_line_heading = 'weighted avg'
    width = max(name_width, len(last_longest_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    dict_result = {}
    all_correct = 0
    all_pred = 0
    all_true = 0
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        all_correct += nb_correct
        all_pred += nb_pred
        all_true += nb_true

        p = 1.0 * nb_correct / nb_pred if nb_pred > 0 else 0
        r = 1.0 * nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)
        dict_result[type_name] = {'precision': p, 'recall': r, 'f1-score': f1, 'support': nb_true}

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    macro_p = 1.0 * all_correct / all_pred if all_pred > 0 else 0
    macro_r = 1.0 * all_correct / all_true if all_true > 0 else 0
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r) if macro_p + macro_r > 0 else 0

    micro_p = float(np.average(ps))
    micro_r = float(np.average(rs))
    micro_f = float(np.average(f1s))

    weighted_p = float(np.average(ps, weights=s))
    weighted_r = float(np.average(rs, weights=s))
    weighted_f = float(np.average(f1s, weights=s))

    avg_support = float(np.sum(s))

    avg_reports = [["micro avg", micro_p, micro_r, micro_f, avg_support],
                   ["macro avg", macro_p, macro_r, macro_f, avg_support],
                   ["weighted avg", weighted_p, weighted_r, weighted_f, avg_support]]

    for i in range(len(avg_reports)):
        report += row_fmt.format(*avg_reports[i],
                                 width=width, digits=digits)
        dict_result[avg_reports[i][0]] = {'precision': avg_reports[i][1], 'recall': avg_reports[i][2],
                                          'f1-score': avg_reports[i][3], 'support': avg_reports[i][4]}

    if dict_report:
        return report, dict_result
    return report
