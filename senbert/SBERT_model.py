# coding=UTF-8

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import models
from torchsummary import summary
import torch


# 设置运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SBERTModel:
    def __init__(self, model_path, max_len=None):
        self.model_path = model_path
        self.max_len = max_len

    def forward(self):
        word_embedding_model = models.Transformer(self.model_path, self.max_len)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,  # 1_Pooling/config.json
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(device)
        # summary(model, input_size=[(512, ), (512, )])

        train_loss = losses.CosineSimilarityLoss(model)

        return model, train_loss
