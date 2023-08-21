# Bert_Sentiment_Analysis_Web

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

Demo地址：
http://134.175.230.159:5000/predict


![效果](https://github.com/mickeyomeow12/text-sentiment-web/blob/master/ezgif.com-video-to-gif.gif)

# 项目说明
此项目用于23春<计算机系统结构>课程实践作业


本项目使用Bert&RoBERTa建模情感分析模型，结合IMDB电影评论数据集，用于探究并预测文本数据背后的情感态度。用于预测任何给定文本的情感为积极或消极。


# 应用实现

## 编程语言及相关技术
前端：JS、HTML、CSS<br>
后端：Python(Flask)<br>
训练深度学习框架：Pytorch<br>
训练环境：GPU2080TI <br>
CPU推理：ONNX模型<br>
服务器：腾讯云CPU2G<br>

## 文件结构
config.py:定义了训练和模型的相关配置，如设备设置、模型输入的最大长度、批处理大小、时代数等。

dataset.py:处理数据集

model.py:PyTorch模块，使用了预训练模型，并添加了一个Dropout层和一个线性输出层。

engine.py:定义了损失函数loss_fn，它使用二元交叉熵损失。提供函数来进行模型的训练。

train.py:集成了配置、数据集、引擎等模块。定义了模型的训练流程，包括数据划分、模型初始化、定义优化器和学习率调度器等。

app.py:Flask应用，用于部署模型并提供API接口。


## 安装依赖
Path:/src

python 相关依赖：

Flask==2.2.5<br> 
numpy==1.19.2<br> 
pandas==1.1.2<br> 
scikit_learn==0.23.2<br> 
torch==1.6.0<br> 
tqdm==4.49.0<br> 
transformers==4.30.2<br> 

## 实现指南

命令行中执行：

```pip install -r requirements.txt```

运行train.py脚本在您的数据集上训练模型。

```python train.py```

运行app.py脚本启动Flask应用。

```python app.py```



# 数据集

IMDB Dataset of 50K Movie Reviews

下载地址：https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

![dataset](https://github.com/mickeyomeow12/text-sentiment-web/blob/master/1.png)


<div align="left">
<img src=https://github.com/mickeyomeow12/text-sentiment-web/blob/master/2.png width=40%/>
</div>

# 模型
## Bert_base_uncased

Bert模型是Google在2018年10月发布的语言模型。BERT 架构由多个堆叠在一起的 Transformer 编码器组成。每个 Transformer 编码器都封装了两个子层：一个自注意力层和一个前馈层。

本项目采用BERT base 模型，由 12 层 Transformer 编码器、12 个注意力头、768 个隐藏大小和 110M 参数组成。在将其用作 BERT 模型的输入之前，需要通过添加 [CLS] 和 [SEP] 标记来对 tokens 的 sequence 重新编码。每个位置输出一个大小为 hidden_ size的向量（BERT Base 中为 768）。使用标准的 PyTorch 训练循环来训练模型。用 Adam 作为优化器，而学习率设置为3e-5。使用分类交叉熵作为我们的损失函数。最后在测试数据上评估模型

首先通过将数据集中的句子转换为一系列tokens (words)即tokenization。

<div align="left">
<img src=https://media.geeksforgeeks.org/wp-content/uploads/20200422012400/Single-Sentence-Classification-Task.png width=40%/>
</div>



## RoBERTa_base

在config.py中切换roberta_base的路径

<div align="left">
<img src=https://production-media.paperswithcode.com/models/roberta-classification.png-0000000936-4dce6670.png width=40%/>
</div>

# 模型效果
| Model             | Accuracy | 
|-------------------|----------|
| Bert_base_uncased | 0.9154   | 
| RoBERTa           | 0.9363   |


分析：Roberta比Bert的效果好，Roberta对于长文本分类确实是Bert进阶




# Demo展示
网页展示效果
![展示](https://github.com/mickeyomeow12/text-sentiment-web/blob/master/demo_1.png)
