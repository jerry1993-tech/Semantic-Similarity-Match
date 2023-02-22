# -*- coding: utf-8 -*-

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
from data_manager import DataManager
from SBERT_model import SBERTModel
from sentence_transformers import InputExample, evaluation
from torch.utils.data import DataLoader
import datetime
import math

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='../senbert/sbert_model_path/hfl_rbt4')
parser.add_argument('--data_name', type=str, default='data')
parser.add_argument('--max_len', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eval_steps', type=int, default=1000)
parser.add_argument('--num_epochs', type=int, default=60)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--stop_after', type=float, default=0.3)
args = parser.parse_args()

model_save_path = '../senbert/sbert_model_path/training_similarity_model_{0}'.format(datetime.datetime.now().strftime(
	"%Y-%m-%d_%H-%M-%S"))


class Train(object):
	# load data and define train examples
	train_seq1_list = DataManager.load_data('../{}/train/train.seq1.in'.format(args.data_name))
	train_seq2_list = DataManager.load_data('../{}/train/train.seq2.in'.format(args.data_name))
	train_label_list = DataManager.load_data('../{}/train/train.label'.format(args.data_name))
	train_dataset = DataManager.create_dataset(train_seq1_list, train_seq2_list, train_label_list)
	valid_seq1_list = DataManager.load_data('../{}/valid/valid.seq1.in'.format(args.data_name))
	valid_seq2_list = DataManager.load_data('../{}/valid/valid.seq2.in'.format(args.data_name))
	valid_label_list = DataManager.load_data('../{}/valid/valid.label'.format(args.data_name))
	valid_dataset = DataManager.create_dataset(valid_seq1_list, valid_seq2_list, valid_label_list)
	test_seq1_list = DataManager.load_data('../{}/test/test.seq1.in'.format(args.data_name))
	test_seq2_list = DataManager.load_data('../{}/test/test.seq2.in'.format(args.data_name))
	test_label_list = DataManager.load_data('../{}/test/test.label'.format(args.data_name))
	test_dataset = DataManager.create_dataset(test_seq1_list, test_seq2_list, test_label_list)

	train_datas = list()
	for i in train_dataset:
		train_datas.append(InputExample(texts=[i[0], i[1]], label=float(i[2])))
	train_data_loader = DataLoader(train_datas, shuffle=True, batch_size=args.batch_size)

	# define valid evaluation examples
	sentences1, sentences2, scores = [], [], []
	for i in valid_dataset:
		sentences1.append(i[0])
		sentences2.append(i[1])
		scores.append(float(i[2]))
	evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

	# Configure the training. Skip evaluation in this example
	warmup_steps = math.ceil(len(train_data_loader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
	# Stopping and Evaluating after 30% of training data (less than 1 epoch)
	# that 20-30% is often ideal for convergence of random seed
	steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size * args.stop_after)
	# load and train the model
	sbert_model = SBERTModel(args.model_path, args.max_len)
	model, train_loss = sbert_model.forward()
	model.fit(train_objectives=[(train_data_loader, train_loss)],
	          evaluator=evaluator,
	          epochs=args.num_epochs,
	          # steps_per_epoch=steps_per_epoch,
	          warmup_steps=warmup_steps,
	          evaluation_steps=args.eval_steps,
	          use_amp=True,
	          optimizer_params={'lr': args.learning_rate},
	          save_best_model=True,
	          output_path=model_save_path)

	# define test evaluation examples
	sentences1, sentences2, scores = [], [], []
	for i in test_dataset:
		sentences1.append(i[0])
		sentences2.append(i[1])
		scores.append(i[2])

	evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
	print(model.evaluate(evaluator))

	evaluator = evaluation.BinaryClassificationEvaluator(sentences1, sentences2, scores)
	print(model.evaluate(evaluator))

	# -------测试训练完的模型-----------
	from sentence_transformers import SentenceTransformer, util
	query_pair = ['我在哪儿购买会员呀', '如何购买会员']

	model_saved = SentenceTransformer(model_save_path)
	print("模型加载成功...")
	embedding_pair = model_saved.encode(query_pair)
	cos_sim = util.pytorch_cos_sim(embedding_pair[0], embedding_pair[1])
	score = cos_sim[0].tolist()
	print(score)


if __name__ == '__main__':
	trainer = Train()
