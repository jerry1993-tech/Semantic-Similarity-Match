#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import json
import yaml
import pathlib
import argparse
import os
import sys
local_dir = os.path.abspath('..')
sys.path.append(local_dir)
from utils.addict import Dict

dir_name = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    def __init__(self, config_dir='config'):
        self.parser = argparse.ArgumentParser(description='model parameters')

        # add command line parameters
        self.parser.add_argument('--model_yaml', type=str, default='ranking_model.yaml')
        self.parser.add_argument('--data_dir', type=str, default='data')
        self.parser.add_argument('--max_sequence_length', type=int, default=None)
        self.parser.add_argument('--batch_size', type=int, default=None)
        self.parser.add_argument('--max_training_epochs', type=int, default=None)
        self.parser.add_argument('--num_gpus', type=int, default=8)
        self.parser.add_argument('--model_dir', type=str, default='saved_model', help='model_dir')

        args, _ = self.parser.parse_known_args()

        self.options = self.load_config(os.path.join(dir_name, '../') + config_dir + '/' + args.model_yaml)

        self.bert_path = os.path.join(dir_name, '../') + self.options.bert_path
        self.teacher_bert_path = os.path.join(dir_name, '../') + self.options.teacher_bert_path
        if self.options.use_slimmed_albert:
            self.bert_path = self.bert_path + "_slim"
            self.options.variable_mapping_file = str(list(pathlib.Path(self.bert_path).glob('variable_name_mapping.json'))[0])
        else:
            self.options.variable_mapping_file = str(list(pathlib.Path(self.bert_path).glob('variable_name_mapping.json'))[0])

        if self.options.use_bert:

            self.options.bert_config_file = str(list(pathlib.Path(self.bert_path).glob('*_config_{}.json'.format(self.options.bert_layer_type)))[0])
            self.options.bert_checkpoint_file = str(list(pathlib.Path(self.bert_path).glob('*.ckpt.index'))[0]).replace('.index', '')

            self.options.bert_config = BertConfig.from_json_file(self.options.bert_config_file)

        if self.options.use_bert and self.options.use_distill:
            self.options.teacher_bert_config_file = str(
                list(pathlib.Path(self.teacher_bert_path).glob('*_config_{}.json'.format(self.options.teacher_bert_layer_type)))[0])
            self.options.teacher_bert_checkpoint_file = str(list(pathlib.Path(self.teacher_bert_path).glob('*.ckpt.index'))[0]).replace(
                '.index', '')

            self.options.teacher_bert_config = BertConfig.from_json_file(self.options.teacher_bert_config_file)

        for key, value in args.__dict__.items():
            if value is not None:
                self.update(self.options, key, value)
                # print('update {} to {} from command line'.format(key, value))

        print(f"{'Applying Parameters':=^100}\n")
        for key, value in self.options.items():
            print('{0:<50}:{1:<100}\n'.format(str(key), str(value)))

    @staticmethod
    def load_config(config_file):
        return Dict(yaml.load(open(config_file), Loader=yaml.FullLoader))

    def update(self, options, key, value):
        if key.find('.') >= 0:
            arr = key.split('.', 1)
            self.update(options[arr[0]], arr[1], value)
        else:
            if isinstance(value, list):
                options[key] = list(filter(lambda x: x != '', value))
            elif value == '':
                options[key] = None
            else:
                options[key] = value


class BertConfig(object):
    """Configuration for `BertModel`."""
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 backward_compatible=True):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
          backward_compatible: Boolean, whether the variables shape are compatible
            with checkpoints converted from TF 1.x BERT.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.backward_compatible = backward_compatible

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class AlbertConfig(BertConfig):
    """Configuration for `ALBERT`."""
    def __init__(self, embedding_size, inner_group_num=1, **kwargs):
        """Constructs AlbertConfig.

        Args:
          embedding_size: Size of the factorized word embeddings.
          inner_group_num: Number of inner repetition of attention and ffn.
          **kwargs: The remaining arguments are the same as above 'BertConfig'.
        """
        super(AlbertConfig, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.inner_group_num = inner_group_num

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `AlbertConfig` from a Python dictionary of parameters."""
        config = AlbertConfig(embedding_size=None, vocab_size=None)
        for (key, value) in json_object.items():
            config.__dict__[key] = value
        return config
