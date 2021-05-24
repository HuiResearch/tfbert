# -*- coding: UTF-8 -*-
# __author__ = 'huanghui'
# __date__ = '2021/5/16 22:16'
# __project__ = 'tfbert'

"""
dureader2021 的评估函数
"""
from __future__ import print_function
from collections import OrderedDict
import io
import json
import six
import sys

if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')
import argparse


def _tokenize_chinese_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def _normalize(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
        u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
        u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > max_len:
                    max_len = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - max_len:p], max_len


def evaluate(ref_ans, pred_ans, verbose=False):
    """
    ref_ans: reference answers, dict
    pred_ans: predicted answer, dict
    return:
        f1_score: averaged F1 score
        em_score: averaged EM score
        total_count: number of samples in the reference dataset
        skip_count: number of samples skipped in the calculation due to unknown errors
    """
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for query_id, sample in ref_ans.items():
        total_count += 1
        para = sample['para']
        query_text = sample['question']
        title = sample['title']
        answers = sample['answers']
        is_impossible = sample['is_impossible']
        try:
            prediction = pred_ans[str(query_id)]
        except:
            skip_count += 1
            if verbose:
                print("para: {}".format(para))
                print("query: {}".format(query_text))
                print("ref: {}".format('#'.join(answers)))
                print("Skipped")
                print('----------------------------')
            continue
        if is_impossible:
            if prediction.lower() == 'no answer':
                _f1 = 1.0
                _em = 1.0
            else:
                _f1 = 0.0
                _em = 0.0
        else:
            _f1 = calc_f1_score(answers, prediction)
            _em = calc_em_score(answers, prediction)
        f1 += _f1
        em += _em
        if verbose:
            print("para: {}".format(para))
            print("query: {}".format(query_text))
            print("title: {}".format(title))
            print("ref: {}".format('#'.join(answers)))
            print("cand: {}".format(prediction))
            print("score: {}".format(_f1))
            print('----------------------------')

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = _tokenize_chinese_chars(_normalize(ans))
        prediction_segs = _tokenize_chinese_chars(_normalize(prediction))
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        prec = 1.0 * lcs_len / len(prediction_segs)
        rec = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * prec * rec) / (prec + rec)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _normalize(ans)
        prediction_ = _normalize(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def read_mrc_dataset(filename, tag=None):
    dataset = OrderedDict()
    with io.open(filename, encoding='utf-8') as fin:
        mrc_dataset = json.load(fin)
    for document in mrc_dataset['data']:
        for paragraph in document['paragraphs']:
            para = paragraph['context'].strip()
            title = ''
            if 'title' in paragraph:
                title = paragraph['title']
            for qa in (paragraph['qas']):
                query_id = qa['id']
                query_text = qa['question'].strip()
                answers = [a['text'] for a in qa['answers']]
                if tag is not None:
                    if not qa['type'].startswith(tag):
                        continue
                is_impossible = False
                if 'is_impossible' in qa:
                    is_impossible = qa['is_impossible']
                if is_impossible:
                    answers = ['no answer']
                dataset[query_id] = {
                    'answers': answers,
                    'question': query_text,
                    'para': para,
                    'is_impossible': is_impossible,
                    'title': title
                }
    return dataset


def read_model_prediction(filename):
    with io.open(filename, encoding='utf-8') as fin:
        model_prediction = json.load(fin)
    return model_prediction


def metric(predictions, gold_file: str, dict_report=False):
    """
    dureader使用的评估函数
    :param predictions: 预测结果（字典）或者预测文件地址
    :param gold_file: 标准答案数据文件
    :param dict_report: 是否返回字典形式结果
    :return:
    """
    ref_ans = read_mrc_dataset(gold_file, tag=None)
    if isinstance(predictions, str):
        pred_ans = json.load(io.open(predictions, encoding='utf-8'))
    elif isinstance(predictions, dict):
        pred_ans = predictions
    else:
        raise ValueError("Please input the file name or prediction result")

    F1, EM, TOTAL, SKIP = evaluate(ref_ans, pred_ans, False)
    output_result = OrderedDict()
    output_result['F1'] = F1
    output_result['EM'] = EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    report = json.dumps(output_result, ensure_ascii=False, indent=4)
    if dict_report:
        return report, output_result
    return report
