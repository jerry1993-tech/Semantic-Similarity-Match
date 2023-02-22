# -*- coding: utf-8 -*-

import os
import json
import tqdm
import time
import torch
from sentence_transformers import SentenceTransformer, util


# fine-tune好的模型地址
finetuned_model_path = './sbert_model_path/training_similarity_model_2022-12-12_11-17-19'

modules_json_path = os.path.join(finetuned_model_path, 'modules.json')
with open(modules_json_path) as fIn:
	modules_config = json.load(fIn)

tf_from_s_path = os.path.join(finetuned_model_path, modules_config[0].get('path'))
print(tf_from_s_path)

# fine-tune 训练时的超参数
max_seq_length = 38
max_query_length = 38

# Enable overwrite to export onnx model and download latest script each time when running this notebook.
enable_overwrite = True
# Total samples to inference. It shall be large enough to get stable latency measurement.
total_samples = 1000

cache_dir = os.path.join(".", "cache_models")
print(cache_dir)
if not os.path.exists(cache_dir):
	os.makedirs(cache_dir)


# Load pretrained model and tokenizer
from transformers import (AutoConfig, AutoModel, AutoTokenizer)

# Load pretrained model and tokenizer
config_class, model_class, tokenizer_class = (AutoConfig, AutoModel, AutoTokenizer)

config = config_class.from_pretrained(tf_from_s_path, cache_dir=cache_dir)
tokenizer = tokenizer_class.from_pretrained(tf_from_s_path, do_lower_case=True, cache_dir=cache_dir)
model = model_class.from_pretrained(tf_from_s_path, from_tf=False, config=config, cache_dir=cache_dir)


# Get the first example data to run the model and export it to ONNX
st = ['我想问怎么打开车窗呀', '我要打开车窗']
inputs = tokenizer(
	st,
	padding=True,
	truncation='longest_first',
	return_tensors="pt",
	max_length=max_seq_length,
)
print(inputs)

# Export the loaded model
output_dir = os.path.join(".", "onnx_models")
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
export_model_path = os.path.join(output_dir, 'finetuned_sbert_similarity_model.onnx')


# 根据设备资源可选
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set model to inference mode, which is required before exporting the model because some operators behave differently in
# inference and training mode.
model.eval()
model.to(device)

if enable_overwrite or not os.path.exists(export_model_path):
	with torch.no_grad():
		symbolic_names = {0: 'batch_size', 1: 'max_seq_length'}
		torch.onnx.export(model,                        # model being run
		                  args=tuple(inputs.values()),  # model input (or a tuple for multiple inputs)
		                  f=export_model_path,          # where to save the model (can be a file or file-like object)
		                  opset_version=11,             # the ONNX version to export the model to
		                  do_constant_folding=True,     # whether to execute constant folding for optimization
		                  input_names=['input_ids',     # the model's input names
		                               'attention_mask',
		                               'token_type_ids'],
		                  output_names=['last_hidden_state', 'pooler_output'],  # the model's output names
		                  dynamic_axes={'input_ids': symbolic_names,  # variable length axes
		                                'attention_mask': symbolic_names,
		                                'token_type_ids': symbolic_names,
		                                'last_hidden_state': symbolic_names,
		                                'pooler_output': symbolic_names})
		print("Model exported at ", export_model_path)


# ==============Inference ONNX Model with ONNX Runtime============
import onnxruntime

sess_options = onnxruntime.SessionOptions()
# Note that this will increase session creation time, so it is for debugging only.
sess_options.optimized_model_filepath = os.path.join(output_dir, 'finetuned_sbert_similarity_model.onnx')

# Specify providers when you use onnxruntime-gpu for CPU inference.
session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])

latency = []
ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
# ort_inputs = {
#     'input_ids':  data[0].cpu().reshape(1, max_seq_length).numpy(),
#     'input_mask': data[1].cpu().reshape(1, max_seq_length).numpy(),
#     'segment_ids': data[2].cpu().reshape(1, max_seq_length).numpy()
# }
start = time.time()
ort_outputs = session.run(None, ort_inputs)
print(ort_outputs[1])
cos_sim = util.pytorch_cos_sim(ort_outputs[1][0], ort_outputs[1][1])
score = cos_sim[0].tolist()
print(score)
latency.append(time.time() - start)
print("OnnxRuntime cpu Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

# ============Load fine-tune model and tokenizer================
import time
from transformers import (AutoConfig, AutoModel, AutoTokenizer)

# Load pretrained model and tokenizer
config_class, model_class, tokenizer_class = (AutoConfig, AutoModel, AutoTokenizer)
config = config_class.from_pretrained(tf_from_s_path, cache_dir=cache_dir)
tokenizer = tokenizer_class.from_pretrained(tf_from_s_path, do_lower_case=True, cache_dir=cache_dir)
model = model_class.from_pretrained(tf_from_s_path, from_tf=False, config=config, cache_dir=cache_dir)

# Measure the latency. It is not accurate using Jupyter Notebook, it is recommended to use standalone python script.
latency = []
with torch.no_grad():
	start = time.time()
	outputs = model(**inputs)
	latency.append(time.time() - start)
	print(outputs[1])
	cos_sim = util.pytorch_cos_sim(outputs[1][0], outputs[1][1])
	score = cos_sim[0].tolist()
	print(score)
print("PyTorch {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))
