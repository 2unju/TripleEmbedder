import torch.nn as nn
from transformers import AutoModel


class KoSimCT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config["model_path"]).to(config["device"])

    def forward(self, query):
        return self.bert(query).last_hidden_state[:, 0]
