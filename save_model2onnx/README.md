
## 可用于线上部署的model类型
此文件夹提供了 savedModel 转换为 onnx 和 FrozenGraph的脚本

* **qarank2onnx.py**  
  model_dirt 为 savedModel 输入目录  
  output_dir 为 onnx model 输出目录  

* **inferONNX.py**  
  为测试onnx转换成功与否的脚本  

* **qarank2frozenpb.py**  
  为 savedModel 转换 frozenGraph 的脚本  
