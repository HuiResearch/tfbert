# textToy
- 基于tensorflow 1.14 的bert系列预训练模型工具
- 支持多GPU训练，支持pb模型导出，自动剔除adam参数
- 采用dataset 和 string handle配合，可以灵活训练预测

## 背景

本项目主要是修改的google官方代码，只是一个简单的代码整理和修改，所以叫textToy。

用完pytorch后真的觉得好用，但是pytorch在部署上没有tensorflow方便。
tensorflow 2.x的话毛病多多，estimator呢我用得不太习惯，平时也用不到tpu....

因此我就整理了这么一个项目，希望能方便和我情况一样的tf boys。
## 说明


config、tokenizer参考的transformers的实现。

内置有自定义的Trainer，像pytorch一样使用tensorflow1.14，具体使用下边会介绍。

目前内置 [文本分类](examples/classification)、[文本多标签分类](examples/multi_label)、[命名实体识别](examples/ner)例子。

内置的几个例子的数据处理代码都支持多进程处理，实现方式参考的transformers。

随机种子设置还有问题，GPU跑出结果一直在变，还想向知道的大佬请教下。
## 支持模型

bert、electra、albert、nezha、wobert

## requirements
```
tensorflow==1.14
tqdm
jieba
```
目前本项目都是在tensorflow 1.14下实现并测试的，所以最好使用1.14版本，因为内部tf导包都是用的

import tensorflow.compat.v1 as tf

## **使用说明**
#### **Config 和 Tokenizer**
使用方法和transformers一样
```python
from textToy import BertTokenizer, BertConfig

config = BertConfig.from_pretrained('config_path')
tokenizer = BertTokenizer.from_pretrained('vocab_path', do_lower_case=True)

inputs = tokenizer.encode(
'测试样例', text_pair=None, max_length=128, pad_to_max_length=True, add_special_tokens=True)

config.save_pretrained("save_path")
tokenizer.save_pretrained("save_path")
```
#### **Trainer**
```python
import tensorflow.compat.v1 as tf
from textToy import Trainer
from textToy.optimizer import create_optimizer

# 创建dataset
def create_dataset(set_type):
    ...

def get_model_fn():
    # model fn输入为inputs 和 is_training
    # 输出为字典，训练传入loss，需要验证预测传入outputs（列表） 
    def model_fn(inputs, is_training):
        ...
        return {'loss': loss, 'outputs': outputs}

    return model_fn


train_dataset, num_train_batch = create_dataset('train')
dev_dataset, num_dev_batch = create_dataset('dev')

# 创建dataset输入的types和shapes
# 下面是分类的types和shapes
# 当然也可以直接从 from textToy.data.classification import return_types_and_shapes
output_types = {"input_ids": tf.int32,
                "input_mask": tf.int32,
                "token_type_ids": tf.int32,
                'label_ids': tf.int64}
output_shapes = {"input_ids": tf.TensorShape([None, None]),
                 "input_mask": tf.TensorShape([None, None]),
                 "token_type_ids": tf.TensorShape([None, None]),
                 'label_ids': tf.TensorShape([None])}

# 配置trainer，传入模型类型 bert、albert、electra、albert、nezha、wobert
# 指定device为 gpu 或 cpu，默认的gpu
trainer = Trainer('bert', output_types, output_shapes, device='gpu')

trainer.build_model(get_model_fn())

t_total = num_train_batch * epochs // gradient_accumulation_steps

# 训练的话创建train_op
# 使用多卡训练器MultiDeviceTrainer时，创建train_op需要传入trainer的gradients和variables
# 因为这里的梯度在训练器中已经自动求多卡的平均了。
train_op = create_optimizer(
    init_lr=learning_rate,
    gradients=trainer.gradients,
    variables=trainer.variables,
    num_train_steps=t_total,
    num_warmup_steps=t_total * 0.1)   

# 配置优化节点
trainer.compile(train_op, max_checkpoints=1)

# 将dataset传入trainer
trainer.build_handle(train_dataset, 'train')
trainer.build_handle(dev_dataset, 'dev')

# 预训练模型加载参数，若不加载，可以调用trainer.init_variables()
trainer.from_pretrained('model_dir')

"""
接下来可以调用 trainer.backward()  梯度累积，会返回loss
              trainer.train_step() 优化参数，不返回
              trainer.eval_step()  会返回loss、model_fn定义的outputs
              trainer.test_step()  返回model_fn定义的outputs
进行训练、验证、预测

"""
```
多卡运行方式，需要设置环境变量CUDA_VISIBLE_DEVICES，内置trainer会读取参数：
```
CUDA_VISIBLE_DEVICES=1,2 python run.py
```
详细例子查看[examples](examples)

#### **export to pb**
查看[examples\classification\export.py](examples/classification/export.py)例子


## **更新记录**

- 2020年9月23日 增加梯度累积，采用trainer.backward(), trainer.zero_grad(), trainer.train_step() 一同进行训练，参考pytorch训练方式。
- 2020年9月21日 第一次上传，支持模型bert、albert、electra、nezha、wobert。

**Reference**  
1. [Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch. ](https://github.com/huggingface/transformers)
2. [TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)
3. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://github.com/google-research/albert)
4. [NEZHA-TensorFlow](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)
5. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://github.com/google-research/electra)
6. [基于词颗粒度的中文WoBERT](https://github.com/ZhuiyiTechnology/WoBERT)
