# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: multi_label.py
@date: 2020/09/15
"""


def multi_label_metric(y_true, y_pred, label_list, dict_report=False):
    '''
    多标签文本分类的评估函数
    :param y_true: 正确标签, one hot 类型，[[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]]
    :param y_pred: 预测标签，one hot 类型，同上
    :param label_list: 标签列表，顺序对应one hot 位置
    :param dict_report:
    :return:
    '''

    def get_value(res):
        if res["TP"] == 0:
            if res["FP"] == 0 and res["FN"] == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
            recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    result = {}
    for i in range(len(label_list)):
        result[i] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for true, pred in zip(y_true, y_pred):
        for a in range(len(label_list)):
            in1 = pred[a] == 1
            in2 = true[a] == 1
            if in1:
                if in2:
                    result[a]["TP"] += 1
                else:
                    result[a]["FP"] += 1
            else:
                if in2:
                    result[a]["FN"] += 1
                else:
                    result[a]["TN"] += 1

    final_result = {}

    # 格式化输出
    headers = ["precision", "recall", "f1-score"]
    target_names = ['%s' % l for l in label_list]
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    longest_last_line_heading = 'micro macro avg'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), 4)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'

    y = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    sumf = 0
    sump = 0
    sumr = 0
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + '\n'
    for i, label in enumerate(label_list):
        p, r, f = get_value(result[i])
        final_result[label] = {"precision": p, "recall": r, "f1-score": f}  # 每个类别下的p r f值
        report += row_fmt.format(*[label, p, r, f], width=width, digits=4)  # 每个类别的string 输出

        sumf += f
        sump += p
        sumr += r
        for z in result[i].keys():
            y[z] += result[i][z]  # 累积总的tp、fp、fn、tn,计算微平均

    report += '\n'

    micro_p, micro_r, micro_f = get_value(y)
    macro_p = sump * 1.0 / len(result)
    macro_r = sumr * 1.0 / len(result)
    macro_f = sumf * 1.0 / len(result)

    average_micro_macro_f = (macro_f + micro_f) / 2.0  # 这是法研杯要素抽取的评价指标，两者平均
    average_micro_macro_p = (macro_p + micro_p) / 2.0
    average_micro_macro_r = (macro_r + micro_r) / 2.0

    final_result['macro avg'] = {"precision": macro_p, "recall": macro_r, "f1-score": macro_f}
    report += row_fmt.format(*['macro avg', macro_p, macro_r, macro_f], width=width, digits=4)

    final_result['micro avg'] = {"precision": micro_p, "recall": micro_r, "f1-score": micro_f}
    report += row_fmt.format(*['micro avg', micro_p, micro_r, micro_f], width=width, digits=4)

    final_result['micro macro avg'] = {"precision": average_micro_macro_p,
                                       "recall": average_micro_macro_r,
                                       "f1-score": average_micro_macro_f}

    report += row_fmt.format(*['micro macro avg', average_micro_macro_p,
                               average_micro_macro_r, average_micro_macro_f], width=width, digits=4)

    if dict_report:
        return report, final_result
    return report
