
# Semantic-Similarity-Match
此模块基于 Sentence-Transformers、pytorch 实现文本匹配

## 1、模块介绍
Sentence-Transformers 是⼀个基于 Transformers 库的句向量表示 Python 框架，内置语义检索以及多种⽂本（对）相似度损失函数，可以较快的实现模型和语义检索，并提供多种预训练的模型，包括中⽂在内的多语⾔模型；  
实际使⽤过程中 Sentence-Transformers 和 Transformers 模型基本互通互⽤，前者多了 Pooling 层（Mean/Max/CLS Pooling），相关论⽂证明 Average Embedding 效果最好。  


## 2、sbert 微调
在句向量获取中可以直接使⽤ Sentence-Transformers 作为编码器，但在特定领域数据上可能需要进⼀步 fine-tune 来获取更好的效果，  
fine-tune 过程主要进⾏⽂本相似度计算任务，亦句对分类任务；此处是为获得更好的句向量，因此使⽤双塔模型（SiameseNetwork ，孪⽣⽹络）微调，⽽⾮常⽤的基于表示的模型 sbert  

主要步骤如下：
- Encoding，使⽤（同⼀个）BERT 分别对 query1 和 query2 进⾏编码 
- Pooling，对最后⼀层进⾏池化操作获得句⼦表示（Mean/Max/CLS Pooling） 
- Computing，计算两个向量的余弦相似度（或其他度量函数），计算 loss 进⾏反向传播

## 3、模型架构
![sbert](https://github.com/xuyingjie521/Semantic-Similarity-Match/blob/main/images/sbert_structrue.jpg)


## 4. 本模块的执行顺序

* **1. load_sbert_model_from_hub.py 下载指定的预训练模型到本地**
* **2. sbert_model_finetune.py 基于指定的预训练模型在领域数据上进一步fine-tune**
* **3. model_predict.py 利用fine-tune的模型进行推理测试**
* **4. sbert2onnx.py  fine-tune的最佳模型转成onnx格式便于线上部署**
