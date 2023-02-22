# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

dir_name = os.path.abspath(os.path.dirname(__file__))


savedModel_dir = os.path.join(dir_name, "../saved_model/trained_model_rbt4_20220929")
output_dir = os.path.join(dir_name, "./frozen_pb_model")

# 定义输入格式
text_input = tf.TensorSpec((None, None), tf.float32, name="text")
type_id_input = tf.TensorSpec((None, None), tf.float32, name="type")

specs = [text_input, type_id_input]

# 加载模型
network = tf.keras.models.load_model(savedModel_dir)

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: network(x))

full_model = full_model.get_concrete_function(specs)

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=output_dir,
                  name="frozen_graph.pb",
                  as_text=False)
