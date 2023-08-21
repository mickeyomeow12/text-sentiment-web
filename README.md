# Bert_Sentiment_Analysis_Web

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

Demo地址：
http://134.175.230.159:5000/predict


![效果](https://github.com/mickeyomeow12/text-sentiment-web/blob/master/ezgif.com-video-to-gif.gif)

# 项目说明
此项目用于23春<计算机系统结构>课程实践作业


本项目使用Bert_base_uncased&RoBERTa建模情感分析模型，结合IMDB电影评论数据集，用于探究并预测文本数据背后的情感态度。用于预测任何给定文本的情感为积极或消极。



# 模型
Bert_base_uncased

<div align="center">
<img src=https://media.geeksforgeeks.org/wp-content/uploads/20200422012400/Single-Sentence-Classification-Task.png  width=200 height=100 />
</div>

![bert]()

RoBERTa_base

![RoBERTa](https://production-media.paperswithcode.com/models/roberta-classification.png-0000000936-4dce6670.png  width=200 height=100)


# 数据集

IMDB Dataset of 50K Movie Reviews

下载地址：https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

![dataset](https://github.com/mickeyomeow12/text-sentiment-web/blob/master/1.png)
![dataset2](https://github.com/mickeyomeow12/text-sentiment-web/blob/master/2.png)

# 模型效果
| Model             | Accuracy | F1     |
|-------------------|----------|--------|
| Bert_base_uncased | 0.9154   | 0.9198 |
| RoBERTa           | 0.9023   | 0.9164 |
|                   |          |        |



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



# Demo展示
网页展示效果
![展示](https://github.com/mickeyomeow12/text-sentiment-web/blob/master/demo_1.png)
