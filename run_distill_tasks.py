# -*- coding: utf-8 -*-

import logging
import random
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from preprocess.config import Config
from preprocess.data import process_data

from nn.callback import PrintBest, common_callbacks, F1Score
from utils import print_metrics
from distill.distillation import Distiller, TeacherModel, StudentModel

tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
print(getattr(tf, '__version__'))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def setup_seed(seed):
	"""设置随机种子"""
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def main(config):
	# 设置分布式训练策略
	strategy = tf.distribute.MirroredStrategy()
	print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
	config.strategy = strategy
	# 数据处理
	process_data(config)
	# dataset
	# batch_size = config.batch_size_per_replica * strategy.num_replicas_in_sync
	# valid_batch_size = config.valid_batch_size * strategy.num_replicas_in_sync
	# test_batch_size = config.test_batch_size * strategy.num_replicas_in_sync
	# train_dataset = config.dataset.train.shuffle(5000).cache().batch(batch_size, drop_remainder=True).prefetch(
	#     strategy.num_replicas_in_sync)
	# valid_dataset = config.dataset.valid.shuffle(5000).cache().batch(valid_batch_size, drop_remainder=True)
	batch_size = config.batch_size_per_replica
	valid_batch_size = config.valid_batch_size
	test_batch_size = config.test_batch_size
	train_dataset = config.dataset.train.shuffle(5000).cache().batch(batch_size, drop_remainder=True)
	valid_dataset = config.dataset.valid.shuffle(5000).cache().batch(valid_batch_size, drop_remainder=True)

	# 可分布式训练模型
	# with strategy.scope():
	teacher, callbacks_t = TeacherModel(config)
	teacher.summary()
	student, callbacks_s = StudentModel(config)
	student.summary()

	# 训练 teacher model
	_ = teacher.fit(
		train_dataset,
		epochs=config.teacher_max_training_epochs,
		validation_data=valid_dataset,
		callbacks=callbacks_t,
		verbose=2)

	result = teacher.evaluate(valid_dataset, verbose=0)
	print_metrics.print_table(teacher.metrics_names, result)

	# 进行蒸馏学习
	distiller = Distiller(student, teacher)

	distiller.compile(optimizer=tfa.optimizers.LazyAdam(learning_rate=config.learning_rate),
	                  metrics=[tf.keras.losses.CategoricalCrossentropy(from_logits=True),
	                           F1Score(),
	                           "accuracy"],
	                  # 由于输出是 logits，所以 from_logits=True来完成 softmax 计算
	                  student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
	                  distill_loss_fn=tf.keras.losses.KLDivergence(),  # loss = y_true * log(y_true / y_pred)
	                  alpha=config.alpha,
	                  temperature=config.temperature)

	# distiller.student.summary()
	# 训练 distiller 中的 student model
	_ = distiller.fit(
		train_dataset,
		epochs=config.max_training_epochs,
		validation_data=valid_dataset,
		callbacks=callbacks_s,
		verbose=2)
	result = distiller.evaluate(valid_dataset, verbose=0)
	print_metrics.print_table(distiller.metrics_names, result)

	# f1-max最佳的实例model来评估测试集
	print("\n\nTest Prediction.\n")
	test_dataset_x = config.dataset.inputs['test']
	test_dataset_y = config.dataset.targets['test']
	text_ids, type_ids = [], []
	for element in test_dataset_x:
		text_id = element[0].numpy()  # shape=(max_len, )
		type_id = element[1].numpy()
		text_ids.append(text_id)  # shape=(sample_num, max_len)
		type_ids.append(type_id)
	input_tokens_ids = tf.convert_to_tensor(np.array(text_ids), dtype=tf.float32)
	input_segments_ids = tf.convert_to_tensor(np.array(type_ids), dtype=tf.float32)
	inputs = [input_tokens_ids, input_segments_ids]
	test_y_logits, test_y_logits_softmax = distiller.student.predict(inputs,
	                                                                 batch_size=test_batch_size,
	                                                                 verbose=1)
	print("test_y_pred=", test_y_logits)
	print("test_y_logits_softmax=", test_y_logits_softmax)

	test_y_true = []
	for y_true in test_dataset_y:
		test_y_true.append(y_true.numpy().tolist())

	print_metrics.multiclass_cal_metrics(test_y_true, test_y_logits_softmax)

	# f1-max最佳模型保存为 save_format='tf':pb格式
	print("\n\nSave Model.\n")
	train_date = datetime.datetime.now().strftime("%Y%m%d")
	saved_model_dir = 'saved_model/trained_model_{0}_{1}'.format(config.bert_layer_type, train_date)
	distiller.student.optimizer = None
	distiller.student.compiled_loss = None
	distiller.student.compiled_metrics = None
	distiller.student.save(saved_model_dir, save_format='tf')


if __name__ == '__main__':
	config = Config().options
	main(config)
