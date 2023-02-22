# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.engine import data_adapter

import os
import sys
local_dir = os.path.abspath('..')
sys.path.append(local_dir)
from preprocess.config import Config

from nn.layer import BertEncoder
from nn.callback import PrintBest, common_callbacks, F1Score
from nn.lr_scheduler import create_learning_rate_scheduler


# Specify the performance metric of Distiller
train_acc = tf.keras.metrics.BinaryAccuracy(name="train_acc")
valid_acc = tf.keras.metrics.BinaryAccuracy(name="valid_acc")


class Distiller(tf.keras.Model):
	__doc__ = """利用tf.keras创建一个Distiller类，
	需要重写tf.keras.Model()的train_step()、test_step()和compile()方法"""

	def __init__(self, student, teacher, *args, **kwargs):
		super(Distiller, self).__init__(*args, **kwargs)
		self.student = student
		self.teacher = teacher

	def compile(self, optimizer, metrics, student_loss_fn, distill_loss_fn, alpha, temperature):
		super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
		self.student_loss_fn = student_loss_fn
		self.distill_loss_fn = distill_loss_fn
		self.alpha = alpha
		self.temperature = temperature

	def train_step(self, data):
		data = data_adapter.expand_1d(data)
		x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

		teacher_logits = self.teacher(x, training=False)  # training=False 教师模型只推理
		with tf.GradientTape() as tape:
			student_logits, _ = self.student(x, training=True)  # training=True 学生模型只训练
			student_loss = self.student_loss_fn(y, student_logits)
			distill_loss = self.distill_loss_fn(tf.nn.softmax(teacher_logits / self.temperature, axis=1),
			                                    tf.nn.softmax(student_logits / self.temperature, axis=1))

			loss = self.alpha * student_loss + (1 - self.alpha) * (self.temperature ** 2) * distill_loss
		trainable_vars = self.student.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		self.compiled_metrics.update_state(y, tf.nn.softmax(student_logits))
		train_acc.update_state(y, tf.nn.softmax(student_logits))
		t_acc = train_acc.result()
		train_acc.reset_states()
		result = {m.name: m.result() for m in self.metrics}
		result.update(
			{"loss": loss, "student_loss": student_loss, "distill_loss": distill_loss, "student_accuracy": t_acc}
		)

		return result

	def test_step(self, data):
		data = data_adapter.expand_1d(data)
		x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

		student_logits, _ = self.student(x, training=False)  # training=False 学生模型只推理
		student_loss = self.student_loss_fn(y, student_logits)

		self.compiled_metrics.update_state(y, tf.nn.softmax(student_logits))
		valid_acc.update_state(y, tf.nn.softmax(student_logits))
		v_acc = valid_acc.result()
		valid_acc.reset_states()
		results = {m.name: m.result() for m in self.metrics}
		results.update({"student_loss": student_loss, "student_accuracy": v_acc})

		return results


def TeacherModel(config):
	__doc__ = """优先使用Functional API创建 teacher 模型实例"""

	# 参数读取
	dropout_keep_prob = config.dropout_keep_prob
	hidden_size = config.hidden_size

	# fine-tune params
	pre_train_trainable = config.pre_train_trainable
	similarity_trainable = config.similarity_trainable
	teacher_learning_rate = config.teacher_learning_rate

	teacher_bert_config = config.teacher_bert_config
	teacher_bert_checkpoint_file = config.teacher_bert_checkpoint_file
	bert_max_sequence_length = teacher_bert_config.max_position_embeddings
	sequence_length = config.max_sequence_length
	max_len = min(bert_max_sequence_length, sequence_length)
	num_classes = config.num_classes

	# 构建 teacher 模型
	text = tf.keras.layers.Input(shape=(max_len,), name="text")  # [None, 2, max_len]
	type_id = tf.keras.layers.Input(shape=(max_len,), name="type")

	# bert modeling
	bert_encoder = BertEncoder.get_bert_encoder(teacher_bert_config, name="BertEncoder")
	bert_encoder.build([[None, None], [None, None]])
	bert_encoder.load_bert(teacher_bert_config, teacher_bert_checkpoint_file)
	bert_encoder.trainable = pre_train_trainable
	ner_logist, cls_output, _ = bert_encoder([text, type_id])

	# similarity
	similarity_ffw0 = tf.keras.layers.Dense(hidden_size,
	                                        activation='relu',
	                                        trainable=similarity_trainable,
	                                        name="similarity_ffw0")(cls_output)
	dropout_out = tf.keras.layers.Dropout(rate=1 - dropout_keep_prob)(similarity_ffw0)
	# 这里的输出是 logits 而非sigmoid(或softmax)激活后的结果
	teacher_logits = tf.keras.layers.Dense(num_classes,
	                                       activation=None,
	                                       trainable=similarity_trainable,
	                                       name="teacher_logits")(dropout_out)

	inputs = [text, type_id]
	outputs = [teacher_logits]

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="teacher_model")

	model.compile(optimizer=tfa.optimizers.LazyAdam(learning_rate=teacher_learning_rate),
	              # 由于输出是 logits，所以 from_logits=True来完成 softmax 计算
	              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
	              metrics=[F1Score(), "accuracy"])

	callbacks = [PrintBest(monitor="val_f1"),
	             tf.keras.callbacks.EarlyStopping(monitor="val_f1",
	                                              patience=config.patience,
	                                              restore_best_weights=True,
	                                              mode="max"),
	             create_learning_rate_scheduler(config, model_type="teacher", verbose=1)
	             ] + common_callbacks

	return model, callbacks


def StudentModel(config):
	__doc__ = """优先使用Functional API创建student模型实例"""

	# 参数读取
	dropout_keep_prob = config.dropout_keep_prob
	hidden_size = config.hidden_size

	# fine-tune params
	pre_train_trainable = config.pre_train_trainable
	similarity_trainable = config.similarity_trainable

	bert_config = config.bert_config
	bert_checkpoint_file = config.bert_checkpoint_file
	bert_max_sequence_length = bert_config.max_position_embeddings
	sequence_length = config.max_sequence_length
	max_len = min(bert_max_sequence_length, sequence_length)
	num_classes = config.num_classes

	# 构建 student 模型
	text = tf.keras.layers.Input(shape=(max_len,), name="text")  # [None, 2, max_len]
	type_id = tf.keras.layers.Input(shape=(max_len,), name="type")

	# bert modeling
	bert_encoder = BertEncoder.get_bert_encoder(bert_config, name="BertEncoder")
	bert_encoder.build([[None, None], [None, None]])
	bert_encoder.load_bert(bert_config, bert_checkpoint_file)
	bert_encoder.trainable = pre_train_trainable
	ner_logist, cls_output, _ = bert_encoder([text, type_id])

	# similarity
	similarity_ffw0 = tf.keras.layers.Dense(hidden_size,
	                                        activation='relu',
	                                        trainable=similarity_trainable,
	                                        name="similarity_ffw0")(cls_output)
	dropout_out = tf.keras.layers.Dropout(rate=1 - dropout_keep_prob)(similarity_ffw0)
	# 这里的输出是 logits 而非sigmoid(或softmax)激活后的结果
	student_logits = tf.keras.layers.Dense(num_classes,
	                                       activation=None,
	                                       trainable=similarity_trainable,
	                                       name="student_logits")(dropout_out)
	# 这里的输出是 logits经softmax激活后的结果，用于predict输出
	student_logits_softmax = tf.keras.layers.Activation('softmax', name='student_logits_softmax')(student_logits)

	inputs = [text, type_id]
	outputs = [student_logits, student_logits_softmax]

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="student_model")

	callbacks = [PrintBest(monitor="val_f1"),
	             tf.keras.callbacks.EarlyStopping(monitor="val_f1",
	                                              patience=config.patience,
	                                              restore_best_weights=True,
	                                              mode="max"),
	             create_learning_rate_scheduler(config, model_type="student", verbose=1),
	             common_callbacks[0]]

	return model, callbacks


if __name__ == "__main__":
	config = Config().options
	teacher, callbacks_t = TeacherModel(config)
	teacher.summary()
	student, callbacks_s = StudentModel(config)
	student.summary()
