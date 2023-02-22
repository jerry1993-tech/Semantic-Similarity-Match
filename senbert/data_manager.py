# -*- coding: utf-8 -*-

import pandas as pd
import logging


# 加载数据
class DataManager(object):
	@staticmethod
	def load_data(filename):
		"""加载数据xx.seq1.in 单条格式：文本1"""
		query_list = []
		with open(filename, encoding='utf-8') as f:
			for l in f:
				try:
					query = l.strip()
					query_list.append(query)
				except ValueError as e:
					logging.info("数据未加载成功！")
			print("len(query_list)=", len(query_list))
		return query_list

	@staticmethod
	def load_labels(filename):
		"""加载数据xx.label 单条格式：int"""
		labels = []
		with open(filename, encoding='utf-8') as f:
			for l in f:
				try:
					l = l.strip()
					labels.append(int(l))
				except ValueError as e:
					logging.info("标签未加载成功！")
		return labels

	@staticmethod
	def load_data_from_csv(csv_file, col_name1, col_name2):
		"""加载csv格式的数据
		:param col_name1: str
		:param col_name2: str
		:return: [[str, str], [str, str], [str, str]...]
		"""
		query_pair_list = []
		df = pd.read_csv(csv_file)
		for index, row in df.iterrows():
			query1 = row[col_name1]
			query2 = row[col_name2]
			query_pair_list.append([query1, query2])
		print("len(query_pair_list)=", len(query_pair_list))
		return query_pair_list

	@staticmethod
	def create_dataset(query1_list, query2_list, labels):
		dataset = []
		for query1, query2, label in zip(query1_list, query2_list, labels):
			dataset.append((query1, query2, int(label)))

		return dataset
