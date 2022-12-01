import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import sentence_to_triple


class TEDataset(Dataset):
    def __init__(self, config):
        def tokenize(q, r=None, o=None):
            if not q:
                return None
            if r and o:
                q = q.strip() + "[SEP]" + r.strip() + "[SEP]" + o.strip()
            return config["tokenizer"](q, add_special_tokens=True, max_length=config["max_len"],
                                       padding='max_length', return_tensors="pt", truncation=True)["input_ids"].to(config["device"])
        data = pd.read_csv(config["data_path"], sep="\t", encoding="utf-8", names=["query", "positive", "negative"])
        data = data.values.tolist()

        self.query = []
        self.positive = []
        self.negative = []
        for q, p, n in tqdm(data, desc="[Load Data]"):
            p = sentence_to_triple(p)
            n = sentence_to_triple(n)
            self.query.append(tokenize(q)[0])
            self.positive.append(tokenize(p[0], p[1], p[2])[0])
            self.negative.append(tokenize(n[0], n[1], n[2])[0])

    def __getitem__(self, idx):
        return self.query[idx], self.positive[idx], self.negative[idx]

    def __len__(self):
        return len(self.query)


class TEVDataset(Dataset):
    def __init__(self, config):
        def tokenize(q, r=None, o=None):
            if not q:
                return None
            if r and o:
                q = q.strip() + "[SEP]" + r.strip() + "[SEP]" + o.strip()
            return config["tokenizer"](q, add_special_tokens=True, max_length=config["max_len"],
                                       padding='max_length', return_tensors="pt", truncation=True)["input_ids"].to(config["device"])
        data = pd.read_csv(config["data_path"], sep="\t", encoding="utf-8", names=["query", "positive", "pos_score"])
        data = data.values.tolist()

        self.query = []
        self.triple = []
        self.score = []
        for q, p, ps in tqdm(data, desc="[Load Data]"):
            p = sentence_to_triple(p)
            self.query.append(tokenize(q)[0])
            self.triple.append(tokenize(p[0], p[1], p[2])[0])
            self.score.append(float(ps))

    def __getitem__(self, idx):
        return self.query[idx], self.triple[idx], self.score[idx]

    def __len__(self):
        return len(self.query)
