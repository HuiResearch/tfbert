# 多标签文本分类使用例子
- [run_ptm.py](run_ptm.py) 是预训练模型例子
- 训练数据是法研杯2019要素抽取中的divorce类别数据集
## 使用
单卡训练
```
CUDA_VISIBLE_DEVICES=1 python run_ptm.py
```
多卡训练
```
CUDA_VISIBLE_DEVICES=0,1 python run_ptm.py
```

