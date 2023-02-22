# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer, util
import torch

# 设置运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载模型到本地
MODEL_NAME = 'hfl/rbt4'
model = SentenceTransformer(
	MODEL_NAME,
	device=device,
	cache_folder='../senbert/sbert_model_path')

# 测试下载的模型
sen_pair = ['精准估计新意图的数量十分困难', '可能会按问题的类型进行聚类']
embedding_pair = model.encode(sen_pair)
print("模型下载成功...")
cos_sim = util.pytorch_cos_sim(embedding_pair[0], embedding_pair[1])
score = cos_sim[0].tolist()
print(score)
