# 文本分类使用例子
- [run_ptm.py](examples\classification\run_ptm.py) 是预训练模型例子
- [run_cnn.py](examples\classification\run_cnn.py) 是一个简单的textCNN例子
- [export.py](examples\classification\export.py) 导出pb的样例
- [predict.py](examples\classification\predict.py) 使用trainer调用训练好模型的预测样例

## 使用
单卡训练
```
CUDA_VISIBLE_DEVICES=1 python run_ptm.py
```
多卡训练
```
CUDA_VISIBLE_DEVICES=0,1 python run_ptm.py
```
textCNN例子用的Trainer，不支持多卡，ptm用的MultiDeviceTrainer,支持多卡
