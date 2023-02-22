# 基于tensorflow微调的知识蒸馏

## 环境配置与模型蒸馏背景

#### 环境配置

- 数据集为 QA_corpus
- 框架为:tensorflow 2.5.0 
- Python: 3.9.0

#### 蒸馏背景

- 知识蒸馏（Knowledge Distillation）最早是Hinton 2014年在论文Dislillation the Knowledge in a Neural Network中提出的概念，主要思想是通过教师模型（teacher）来指导学生模型（student）的训练，将复杂、学习能力强的教师模型学到的特征表示“知识蒸馏”出来，传递给参数小、学习能力弱的学生模型，从而得到一个速度快、表达能力强的学生模型。  
随着预训练语言模型的不断发展，虽然一直在各种任务上sota，但实际业务中还需要考虑模型的推理性能，而直接用小模型初始化训练效果又不是很理想，所以知识蒸馏在业务中也逐渐被广泛使用，本项目也基于蒸馏的方式进一步优化模型效果。


## 蒸馏论文要点解读

- **[2014,Hinton,《Dislillation the Knowledge in a Neural Network》](https://arxiv.org/abs/1503.02531)**

> When the correct labels are known for all or some of the transfer set, this method can be significantly improved by also training the distilled model to produce the correct labels.

- 这里说对于迁移集来说，如果能够知道部分或全部的准确的标签的话，其方法就能够借助同时训练学生模型去产生正确的标签从而得到模型的改善
> One way to do this is to use the correct labels to modify the soft targets, but we found that a better way is to simply use a weighted average of two different objective functions. The first objective function is the cross entropy with the soft targets and this cross entropy is computed using the same high temperature in the softmax of the distilled model as was used for generating the soft targets from the cumbersome model. 
- 但是一种更常用的方法是使用两个目标函数(也就是损失函数)的一种加权平均，其中一种损失函数是基于软标签的交叉熵，这种交叉熵的计算是使用和生成这种软标签的大模型一样的温度T

```python
公式：distillation_loss = crossentropy(softmax(teacher_logit/T),softmax(student_logit/T))
但是教师网络给予的标签未必是完全正确的，因此我们使用交叉熵的一个变种KL散度(H(P,Q)=H(P)+KL(P||Q)),由于我们缺乏足够可信的H(P),即教师网络的知识未必完全正确，但是其给出的预测分布所代表的知识却足够可信。因此这里的crossentropy=>KLDivergence
```

> The second objective function is the cross entropy with the correct labels. This is computed using exactly the same logits in softmax of the distilled model but at a temperature of 1. We found that the best results were generally obtained by using a condiderably lower weight on the second objective function. Since the magnitudes of the gradients produced by the soft targets scale as 1/T^2 it is important to multiply them by T^2 when using both hard and soft targets. This ensures that the relative contributions of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed while experimenting with meta-parameters. 

- 这里的第二个损失函数是student_loss, 它是基于正确的标签的交叉熵，通过计算使用精确的得分(设置T=1),并且最好的结果出现在设定相对更低的权重。作者认为被软标签（distillation_loss）所产生的梯度的量级大概会是 1/T^2 ，所以需要给没有除以T的student_loss乘上 T^2 这个系数以补偿。但是从实现来看，这个 1/T^2 很难度量，因此我们仅仅通过实验给予最好的权重系数 alpha ，比如0.1/0.3感觉实验效果都还行


- **[2021,Google Research,《Knowledge distillation:A good teacher is patient and consistent》](https://arxiv.org/abs/2106.05237v2)**

> Distillation loss. We use the KL-divergence between the teacher’s pt, and the student’s ps predicted class probability vectors as a distillation loss, as was originally introduced in [12]. We do not use any additional loss term with respect to the original dataset’s hard labels.

```python
公式：distillation_loss = KLDivergence(softmax(teacher_logit/T),softmax(student_logit/T))
对于hard label，使用KL和CE是一样的，因为KL(P||Q)=H(P||Q)-H(P)，训练集不变时label分布是一定的。但对于soft label则不同了，Hinton采用的CE，而本文采用的KL
我比较赞同使用KL，因为教师网络给予的标签未必是完全正确的，因此我们使用交叉熵的一个变种KL散度(H(P,Q)=H(P)+KL(P||Q)),由于我们缺乏足够可信的H(P),即教师网络的知识未必完全正确，但是其给出的预测分布所代表的知识却足够可信。因此使用KLDivergence更好。
```

- **[2019,Microsoft,Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355)**
- 对bert有关的的知识蒸馏
  ![总结导图](https://github.com/xuyingjie521/Semantic-Similarity-Match/blob/main/images/bert知识蒸馏.png)


## 本项目采取的蒸馏方式

- **采用soft-hard_label distill 和 distillation_loss=KLDivergence**

- **定义三种损失函数如下：**

- hard label: 类似于[1,0,0,0]这样的具有one_hot编码的，soft labels: 由训练过的教师给出，如[0.75,0.15,0.02,0.08]
- teacher_loss = CategoricalCrossEntropy(from_logits=True)

- distillation_loss: 学习学生网络的预测和教师网络预测的分布相似性,由于仅仅衡量对于同一随机变量的分布相似性，不使用ground_truth，使用交叉熵是没有道理的。所以定义这种相似性的损失为KLDivergence,它们是基于软标签做的而且伴随着蒸馏温度T
	
```python
distillation_loss = KLDivergence(softmax(teacher_logit/T),softmax(student_logit/T))
```
	
- student_loss: 损失是学生网络的预测和ground_truth做的，基于传统的硬标签

```python
student_loss = CategoricalCrossEntropy(from_logits=True)
```

- 综合 loss 损失由前面两者加权和，其中 alpha 为调节soft-hard_label的超参数，一般选取小于0.1～0.3从而让分布调节到更接近后者，由于 softed softmax计算时需要除以 T，导致关联的soft_label的梯度幅值被缩小了 T^2 倍，所以在计算loss时考虑乘上 T^2 这个系数以补偿，
loss = alpha * student_loss + (1-alpha) * T^2 * distillation_loss  

 ![蒸馏框架](https://github.com/xuyingjie521/Semantic-Similarity-Match/blob/main/images/distilled-Tiny-Roberta-picture.png)  

- 蒸馏内部的学习过程图解如下：

![图解知识蒸馏](https://github.com/xuyingjie521/Semantic-Similarity-Match/blob/main/images/图解知识蒸馏.jpeg)


