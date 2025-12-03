import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

class FraudDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(list(df["Message"]), truncation=True, padding=True, max_length=128)
        self.labels = list(df["Label"])
        self.risks = list(df["source_risk"])
        self.keywords = list(df["keyword_hits"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["risk"] = torch.tensor([self.risks[idx]], dtype=torch.float)
        item["keywords"] = torch.tensor([self.keywords[idx]], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)