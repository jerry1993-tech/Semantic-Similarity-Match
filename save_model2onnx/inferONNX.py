# -*- coding: utf-8 -*-
import psutil
import onnxruntime
import numpy
import tensorflow as tf
import time
import tf2onnx
import os


tf2onnx.logging.set_level(tf2onnx.logging.ERROR)
dir_name = os.path.abspath(os.path.dirname(__file__))


########## Inference the Exported Model with ONNX Runtime##############
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

seq_max_len = 40
# 测试文本 ids
# [CLS 音量变大一点 SEP 把音量搞大一点 SEP]
text_ori = [101, 7509, 7030, 1359, 1920, 671, 4157, 102, 2828, 7509, 7030, 3018, 1920, 671, 4157, 102,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
_type_ori = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

inputs = {
    "text": tf.expand_dims(tf.convert_to_tensor(text_ori, dtype=tf.float32), 0),
    "type": tf.expand_dims(tf.convert_to_tensor(_type_ori, dtype=tf.float32), 0),
}

# from transformers import (TFBertForQuestionAnswering, BertTokenizer)
#
# cache_dir = "/home/xxx/pretrain_models/bert/bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
# question, text = "What is ONNX Runtime?", "ONNX Runtime is a performance-focused inference engine for ONNX models."
# # Pad to max length is needed. Otherwise, position embedding might be truncated by constant folding.
# inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors='tf',
#                                max_length=512, pad_to_max_length=True, truncation=True)


sess_options = onnxruntime.SessionOptions()

# Set the intra_op_num_threads
sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

# Providers is optional. Only needed when you use onnxruntime-gpu for CPU inference.
session = onnxruntime.InferenceSession(output_model_path, sess_options, providers=['CPUExecutionProvider'])

batch_size = 1
inputs_onnx = {k_: numpy.repeat(v_, batch_size, axis=0) for k_, v_ in inputs.items()}
# inputs_onnx = {k_: numpy.repeat(v_, batch_size, axis=0) for k_, v_ in inputs.items()}

# Warm up with one run.
results = session.run(None, inputs_onnx)
print("results of onnx_model=", results)

# Measure the latency.
start = time.time()
for _ in range(total_runs):
    results = session.run(None, inputs_onnx)
end = time.time()
print("ONNX Runtime cpu inference time for sequence length {} (model not optimized): {} ms".format(seq_max_len, format(
    (end - start) * 1000 / total_runs, '.2f')))
del session
