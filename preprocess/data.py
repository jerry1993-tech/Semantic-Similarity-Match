# -*- coding: utf-8 -*-
import os
import sys
import pathlib
import numpy as np
import tensorflow as tf

local_dir = os.path.abspath('..')
sys.path.append(local_dir)
from preprocess.tokenization import BertTextTokenizer, LabelTokenizer

dir_name = os.path.abspath(os.path.dirname(__file__))

types = ["train", "valid", "test"]


def read_data_from_file(config):
    for feed in config.feeds.keys():
        for _type in types:
            if isinstance(config.feeds[feed].data.path, list):
                data = []
                for path in config.feeds[feed].data.path:
                    data_path = dir_name + '/../' + config.data_dir + '/' + path.format(type=_type)
                    with open(data_path) as f:
                        data.append(f.read().splitlines())
                config.feeds[feed].data[_type] = data
                config.length[_type] = len(config.feeds.encoder_inputs.data[_type][0])
            else:
                data_path = dir_name + '/../' + config.data_dir + '/' + config.feeds[feed].data.path.format(type=_type)
                with open(data_path) as f:
                    data = f.read().splitlines()
                config.feeds[feed].data[_type] = data
                config.length[_type] = len(config.feeds.encoder_inputs.data[_type])

        print(f"read {feed} finish")
    print("read data finish")


def process_feeds(config):

    for feed in config.feeds:
        if config.feeds[feed].data_type == "bert":
            sequence_length = config.max_sequence_length
            bert_max_sequence_length = config.bert_config.max_position_embeddings
            max_len = min(bert_max_sequence_length, sequence_length)

            bert_path = os.path.join(dir_name, '../') + config.bert_path
            bert_vocab_file = str(list(pathlib.Path(bert_path).glob('*.txt'))[0])
            tokenizer = BertTextTokenizer.load_from_vocab_txt(bert_vocab_file)  # 直接从vocab.txt生成tokenizer
            config.feeds[feed].vocabsize = tokenizer.vocabsize
            config.feeds[feed].tokenizer = tokenizer

            for _type in types:
                data = config.feeds[feed].data[_type]
                if len(data) == 2:
                    text_ids, type_ids, first_token_lengths, second_token_lengths = tokenizer.batch_encode(data[0],
                                                                                                           data[1],
                                                                                                           max_len=max_len)

                else:
                    text_ids, type_ids, first_token_lengths, second_token_lengths = tokenizer.batch_encode(data[0],
                                                                                                           max_len=max_len)
                config.feeds[feed].ids[_type] = [text_ids, type_ids]
                # first代表模型输入的前半句，second 为第二句
                config.first_token_lengths[_type] = first_token_lengths
                config.second_token_lengths[_type] = second_token_lengths

        if "label" in config.feeds[feed].data_type:
            data_path = dir_name + '/../' + config.data_dir + '/' + config.feeds[feed].data.path.format(type="train")
            vocab_path = dir_name + '/../' + config.data_dir + '/' + config.feeds[feed].vocab.path

            if "binary" in config.feeds[feed].data_type:
                tokenizer = LabelTokenizer.load_from_txt(data_path, vocab_path, binary=True)
            else:
                tokenizer = LabelTokenizer.load_from_txt(data_path, vocab_path)
            config.feeds[feed].tokenizer = tokenizer
            config.feeds[feed].vocabsize = tokenizer.vocabsize

            for _type in types:
                data = config.feeds[feed].data[_type]
                ids = tokenizer.batch_encode(data)
                # 转one-hot形式：适用于多分类
                from tensorflow.keras.utils import to_categorical
                ids_new = to_categorical(ids, config.num_classes)

                if "binary" in config.feeds[feed].data_type:
                    assert tokenizer.vocabsize == 2
                    config.feeds[feed].ids[_type] = ids_new
                    config.feeds[feed].vocabsize = 1  # 单分类
                else:
                    config.feeds[feed].ids[_type] = ids_new
        print("process {} finish".format(feed))
    print("process feed finish")


def process_dataset(config):
    for _type in types:
        for feed_type in ["inputs", "targets"]:
            _list = []
            for feed_item in config[feed_type]:
                if config.feeds[feed_item].data_type == "bert":
                    _list.append(np.array(config.feeds[feed_item].ids[_type][0]))
                    _list.append(np.array(config.feeds[feed_item].ids[_type][1]))
                else:
                    _list.append(np.array(config.feeds[feed_item].ids[_type]))
            if len(_list) > 1:
                _set = tf.data.Dataset.from_tensor_slices(tuple(_list))
            else:
                _set = tf.data.Dataset.from_tensor_slices(_list[0])
            config.dataset[feed_type][_type] = _set
        config.dataset[_type] = tf.data.Dataset.zip((config.dataset.inputs[_type], config.dataset.targets[_type]))

    for feed_type in ["inputs", "targets"]:
        for feed_item in config[feed_type]:
            config.feeds[feed_item].ids = None
            config.feeds[feed_item].data = None

    print("process dataset finish")


def process_data(config):
    read_data_from_file(config)
    process_feeds(config)
    process_dataset(config)


if __name__ == '__main__':
    from config import Config

    config = Config().options
    process_data(config)