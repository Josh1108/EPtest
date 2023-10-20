import torch.nn as nn
from transformers import AutoModel
from typing import Dict
from eptests.registry import register, ENCODER


@register(_name="default", _type=ENCODER)
class SpanEncoder(nn.Module):
    """
    We have to index each data point separately so this part is inefficient
    """

    def __init__(self, hf_model_or_loc: str, layer_index: int, token_pooler):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_model_or_loc)
        self.token_pooler = token_pooler
        for param in self.model.base_model.parameters():  # freezing encoding layers
            param.requires_grad = False
        self.layer_index = layer_index if layer_index != -1 else self.model.config.num_hidden_layers

    def forward(self, batch: Dict, span_key: str):
        # sentence are id's
        _out = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
        # _out.hidden_states dim -> (num_layers,[batch_size,num_tokens,hidden_dim])
        layer_embs = _out.hidden_states[self.layer_index]
        span = [
            [x + 1 for x in y if x != -1] for y in batch[span_key]
        ]  # 1 added due to CLS token embeddings that was passed during tokenization
        # but handle padding anyway
        span_embs = []
        for i, _spn in enumerate(span):
            span_emb = layer_embs[i, _spn[0] : _spn[-1] + 1, :]
            span_embs.append(span_emb)
        # print("Dimensions of embeddings expected [batch_size,num_tokens_span,hidden_dim]",span_embs.shape)
        pooled_emb = self.token_pooler(span_embs)
        return pooled_emb
