# 实体识别使用例子
- [run_ptm.py](run_ptm.py) 是预训练模型例子
- [export.py](export.py) 导出pb的样例

## 使用
单卡训练
```
CUDA_VISIBLE_DEVICES=1 python run_ptm.py
```
多卡训练
```
CUDA_VISIBLE_DEVICES=0,1 python run_ptm.py
```

