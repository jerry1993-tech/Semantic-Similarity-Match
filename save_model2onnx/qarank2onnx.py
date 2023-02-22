# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import tf2onnx


tf2onnx.logging.set_level(tf2onnx.logging.ERROR)
dir_name = os.path.abspath(os.path.dirname(__file__))
print(dir_name)

#######################
savedModel_dir = os.path.join(dir_name, "../saved_model/trained_model_rbt4_20220929")
model_name_or_path = "qarank"
output_dir = os.path.join(dir_name, "./onnx_model")
output_model_path = os.path.join(output_dir, 'tf2onnx_{}.onnx'.format(model_name_or_path))
opset_version = 13
use_external_data_format = False
# Whether allow overwrite existing script or model.
enable_overwrite = False
# Number of runs to get average latency.
total_runs = 100

#######################
# from transformers import (TFBertForQuestionAnswering, BertTokenizer)4

# cache_dir = "/home/xxx/pretrain_models/bert/bert-base-uncased"
# model_name_or_path = 'bert-large-uncased-whole-word-masking-finetuned-squad'
# model_name_or_path = "bert-base-cased"
# is_fine_tuned = (model_name_or_path == 'bert-large-uncased-whole-word-masking-finetuned-squad')

# Load model and tokenizer
# tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
# model = TFBertForQuestionAnswering.from_pretrained(model_name_or_path, cache_dir=cache_dir)
# Needed this to export onnx model with multiple inputs with TF 2.2
# model._saved_model_inputs_spec = None
########################################

model = tf.keras.models.load_model(savedModel_dir, compile=False)

specs = []
text_input = tf.TensorSpec((None, None), tf.float32, name="text")
type_id_input = tf.TensorSpec((None, None), tf.float32, name="type")


specs.append(text_input)
specs.append(type_id_input)


# 转换onnx
if enable_overwrite or not os.path.exists(output_model_path):
    start = time.time()
    _, _ = tf2onnx.convert.from_keras(model,
                                      input_signature=tuple(specs),
                                      opset=opset_version,
                                      large_model=use_external_data_format,
                                      output_path=output_model_path)
    print("tf2onnx run time = {} s".format(format(time.time() - start, '.2f')))

