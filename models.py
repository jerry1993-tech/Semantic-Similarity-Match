# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from nn.loss import CategoricalCrossentropyWithMask, SparseCategoricalCrossentropyWithMask
from nn.metric import F1ScoreWithMask, SparseF1ScoreWithMask
from nn.callback import EarlyStoppingWithWeightF1, PrintWeightF1, PrintBest, WeightF1, common_callbacks, F1Score
from nn.layer import EmbeddingUnique, DownSample, TwoDAttention, PositionalEncoding, TransformerEncoderLayer, \
    AlbertEncoder, ScaledDotProductAttention, BertEncoder
from nn.lr_scheduler import create_learning_rate_scheduler
from tensorflow.keras.optimizers import Adam


def similarity_model(config):
    # 参数读取
    dropout_keep_prob = config.dropout_keep_prob
    hidden_size = config.hidden_size

    # fine tune params
    pre_train_trainable = config.pre_train_trainable
    similarity_trainable = config.similarity_trainable
    learning_rate = config.learning_rate

    bert_config = config.bert_config
    bert_checkpoint_file = config.bert_checkpoint_file
    bert_max_sequence_length = bert_config.max_position_embeddings
    sequence_length = config.max_sequence_length
    max_len = min(bert_max_sequence_length, sequence_length)
    num_classes = config.num_classes

    # 构建模型
    text = tf.keras.layers.Input(shape=(max_len, ), name="text")  # [None, 2, max_len]
    type_id = tf.keras.layers.Input(shape=(max_len, ), name="type")

    # bert Semantic modeling
    bert_encoder = BertEncoder.get_bert_encoder(bert_config, name="BertEncoder")
    bert_encoder.build([[None, None], [None, None]])
    bert_encoder.load_bert(bert_config, bert_checkpoint_file)
    bert_encoder.trainable = pre_train_trainable
    ner_logist, cls_output, _ = bert_encoder([text, type_id])

    # similarity
    similarity = tf.keras.layers.Dense(hidden_size,
                                       activation='relu',
                                       trainable=similarity_trainable,
                                       name="similarity_ffw0")(cls_output)
    similarity = tf.keras.layers.Dropout(rate=1 - dropout_keep_prob)(similarity)
    similarity = tf.keras.layers.Dense(num_classes,
                                       activation='sigmoid',
                                       trainable=similarity_trainable,
                                       name="similarity")(similarity)

    inputs = [text, type_id]
    outputs = [similarity]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    loss = {"similarity": tf.keras.losses.BinaryCrossentropy()}
    metric = {"similarity": [tf.keras.metrics.BinaryCrossentropy(),
                             F1Score()]}

    optimizer = tfa.optimizers.LazyAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metric)
    model.summary()

    callbacks = [PrintBest(monitor="val_f1"),
                 tf.keras.callbacks.EarlyStopping(monitor="val_f1",
                                                  patience=config.patience,
                                                  restore_best_weights=True,
                                                  mode="max"),
                 create_learning_rate_scheduler(config.learning_rate, model_type="student", verbose=1)
                 ] + common_callbacks

    class_weight = None

    return model, callbacks, class_weight


if __name__ == "__main__":
    from preprocess.config import Config

    config = Config().options
    for i in config.feeds:
        config.feeds[i].vocabsize = 2
    model = similarity_model(config)
