import torch
import torch.nn as nn
from typing import List
from eptests.registry import register, TOKEN_POOLER, SPAN_POOLER


@register(_name="mean", _type=SPAN_POOLER)
class MeanSpanPooler(nn.Module):
    def __init__(self):
        super().__init__()

    # flake8: noqa
    # noinspection PyMethodMayBeStatic
    def forward(self, span1, span2):
        # [batch_size,hidden_dim]
        span_pooled = torch.mean(torch.stack((span1, span2)), dim=0)
        # print("span_pooled dimensions:",span_pooled.shape)
        return span_pooled


@register(_name="mean", _type=TOKEN_POOLER)
class MeanTokPooler(nn.Module):
    def __init__(self):
        super().__init__()

    # flake8: noqa
    # noinspection PyMethodMayBeStatic
    def forward(self, embedding: List[torch.tensor]):
        # [batch_size,tensor(tok_length,hidden_dim)]
        mean_embedding = []
        for emb in embedding:
            mean_embedding.append(torch.mean(emb, dim=0))

        # print("mean_embedding dimensions:",len(mean_embedding),len(mean_embedding[0]))
        return torch.stack(mean_embedding)
