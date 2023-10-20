import torch.nn as nn
import torch.nn.functional as F
from eptests.registry import CLASSIFIER, register


class ClassifierLayer(nn.Module):
    def __init__(self, hsz, num_labels, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.hsz = hsz
        self.dropout = nn.Dropout(0.1)


@register(_name="linear", _type=CLASSIFIER)
class Linear(ClassifierLayer):
    def __init__(self, hsz, num_labels, **kwargs):
        super().__init__(hsz, num_labels)
        self.classifier = nn.Linear(self.hsz, self.num_labels)

    def forward(self, span_embeddings):
        return self.classifier(self.dropout(span_embeddings))


@register(_name="mlp", _type=CLASSIFIER)
class Mlp(ClassifierLayer):
    def __init__(self, hsz, num_labels, **kwargs):
        super().__init__(hsz, num_labels)
        self.hidden_fc = nn.Linear(self.hsz, 1024)
        self.classifier = nn.Linear(1024, self.num_labels)

    def forward(self, span_embeddings):
        return self.classifier(F.relu(self.hidden_fc(self.dropout(span_embeddings))))


@register(_name="mlp3", _type=CLASSIFIER)
class Mlp3(ClassifierLayer):
    def __init__(self, hsz, num_labels, **kwargs):
        super().__init__(hsz, num_labels)
        self.hidden_fc2 = nn.Linear(self.hsz, 1024)
        self.hidden_fc1 = nn.Linear(self.hsz, self.hsz)
        self.classifier = nn.Linear(1024, self.num_labels)

    def forward(self, span_embeddings):
        return self.classifier(F.relu(self.hidden_fc2(F.relu(self.hidden_fc1(self.dropout(span_embeddings))))))
