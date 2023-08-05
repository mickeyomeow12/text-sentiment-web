# Bert_Sentiment_Analysis_Web

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

Demo地址：

# 项目说明
本项目使用Bert_base_uncased建模情感分析模型，结合IMDB电影评论数据集，用于探究并预测文本数据背后的情感态度。

此项目用于23春<计算机系统结构>课程实践作业

# 模型
Bert_base_uncased<br>
下载地址:https://huggingface.co/bert-base-uncased

Path:/input

# 数据集

IMDB Dataset of 50K Movie Reviews

下载地址：https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Path:/input

# 模型效果



# 安装依赖
Path:/src
此项目可以用于预测任何给定文本的情感。

python 相关依赖：

Flask==2.2.5<br> 
numpy==1.19.2<br> 
pandas==1.1.2<br> 
scikit_learn==0.23.2<br> 
torch==1.6.0<br> 
tqdm==4.49.0<br> 
transformers==4.30.2<br> 

命令行中执行：

```pip install -r requirements.txt```

运行train.py脚本在您的数据集上训练BERT模型。

```python train.py```

运行app.py脚本启动Flask应用。
```python app.py```

# Demo展示
网页展示效果

