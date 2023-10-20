import torch
from typing import Dict
from baseline.pytorch.classify.model import ClassifierModelBase
from eptests.core.utils import get_device
from eptests.registry import register, PROBE
from eptests.registry import ENCODER_REGISTRY, TOKEN_POOLER_REGISTRY, SPAN_POOLER_REGISTRY, CLASSIFIER_REGISTRY
from eptests.core.utils import remove_key_from_dict
from eptests.datamodels import SPAN1, SPAN2


@register(_name="one-span", _type=PROBE)
class OneSpanProbe(ClassifierModelBase):
    """
    This works when there are 2 spans. Make edits to make it work for just one span. remove pooled_s and pooled_span2.
    """

    def __init__(self, multi_label: bool, **kwargs):
        super().__init__()
        self.multi_label = multi_label
        encoder_params = kwargs["encoder"]
        token_pooler_params = kwargs["token_pooler"]
        classifier_params = kwargs["classifier"]
        self.token_pooler = TOKEN_POOLER_REGISTRY[token_pooler_params["name"]](
            **remove_key_from_dict(token_pooler_params)
        )
        self.encoder = ENCODER_REGISTRY[encoder_params["name"]](
            token_pooler=self.token_pooler, **remove_key_from_dict(encoder_params)
        )
        self.classifier = CLASSIFIER_REGISTRY[classifier_params["name"]](**remove_key_from_dict(classifier_params))

    def forward(self, batch: Dict):
        return self.classifier(self.encoder(batch, span_key=SPAN1))

    def make_input(self, batch_dict: Dict, perm=False, numpy_to_tensor=True):
        return {k: v.to(get_device()) for k, v in batch_dict.items()}

    def create_loss(self):
        if self.multi_label:
            return torch.nn.BCEWithLogitsLoss()
        return torch.nn.CrossEntropyLoss()


@register(_name="two-span", _type=PROBE)
class TwoSpanProbe(OneSpanProbe):
    """ """

    def __init__(self, multi_label: bool, **kwargs):
        super().__init__(multi_label, **kwargs)
        span_pooler_params = kwargs["span_pooler"]
        self.span_pooler = SPAN_POOLER_REGISTRY[span_pooler_params["name"]](**remove_key_from_dict(span_pooler_params))

    def forward(self, batch):
        pooled_span1_embedding = self.encoder(batch, span_key=SPAN1)  # e.g. bert, roberta
        pooled_span2_embedding = self.encoder(batch, span_key=SPAN2)
        pooled_s = self.span_pooler(pooled_span1_embedding, pooled_span2_embedding)
        # e.g. attention based or mean of embeddings
        return self.classifier(pooled_s)
