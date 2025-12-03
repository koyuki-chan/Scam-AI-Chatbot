import torch.nn as nn
from transformers import BertModel
import torch
class FraudClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, risk, keywords):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat((cls_output, risk, keywords), dim=1)
        x = self.dropout(combined)
        return self.classifier(x)