# -*- coding: utf-8 -*-

from data_manager import DataManager
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os
import sys
os.chdir(sys.path[0])
sys.path.append("..")
from utils import print_metrics


class Predict(object):

	def __init__(self, model_save_path):
		self.model = SentenceTransformer(model_save_path)
		print("模型加载成功&推理中...")

	def predict(self, dataset):
		"""
		:param dataset: [(query1, query2, label),...] or [[query1, query2, label],...]
		or [(query1, query2),...] or [[query1, query2],...]
		"""
		labels = []
		pred_scores = []
		if len(dataset[0]) >= 3:
			for data in dataset:
				query_pair = [data[0], data[1]]
				labels.append(data[2])
				embedding_pair = self.model.encode(query_pair)
				cos_sim = util.pytorch_cos_sim(embedding_pair[0], embedding_pair[1])
				score = cos_sim[0].tolist()
				pred_scores.append(score)
			pred_scores_arr = np.array(pred_scores)
			# 评估结果打印
			print_metrics.binary_cal_metrics(labels, pred_scores_arr)
		else:
			for data in dataset:
				query_pair = [data[0], data[1]]
				embedding_pair = self.model.encode(query_pair)
				cos_sim = util.pytorch_cos_sim(embedding_pair[0], embedding_pair[1])
				score = cos_sim[0].tolist()
				pred_scores.append(score)

		return pred_scores

	def predict_single(self, query_pair):
		embedding_pair = self.model.encode(query_pair)
		cos_sim = util.pytorch_cos_sim(embedding_pair[0], embedding_pair[1])
		score = cos_sim[0].tolist()

		return score


if __name__ == '__main__':
	model_save_path = '../senbert/sbert_model_path/training_similarity_model_2022-12-12_11-17-19'
	# 加载 test 数据
	data_name = 'data'
	test_seq1_list = DataManager.load_data('../{}/test/test.seq1.in'.format(data_name))
	test_seq2_list = DataManager.load_data('../{}/test/test.seq2.in'.format(data_name))
	test_label_list = DataManager.load_data('../{}/test/test.label'.format(data_name))
	test_dataset = DataManager.create_dataset(test_seq1_list, test_seq2_list, test_label_list)

	predictor = Predict(model_save_path)
	predictor.predict(test_dataset)
	query_pair = ['我在哪儿购买会员呀', '如何购买会员']
	score = predictor.predict_single(query_pair)
	print(score)
