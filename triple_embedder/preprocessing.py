import re
import torch.nn as nn
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from mapping import Mapping


class SimilarityScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config["model_path"]).to(config["device"])
        self.cos = config["cos"]

    def forward(self, query, candidate):
        query_embed, _ = self.bert(**query, return_dict=False)
        query_embed = query_embed[:, 0]
        knowledge_embed, _ = self.bert(**candidate, return_dict=False)
        knowledge_embed = knowledge_embed[:, 0]

        if query["input_ids"].shape[0] != 1:
            query_embed = query_embed.unsqueeze(1)
            knowledge_embed = knowledge_embed.unsqueeze(0)
        q_sim = self.cos(query_embed, knowledge_embed)
        return q_sim


def list_to_triple(tlist):
    for i, word in enumerate(tlist):
        if (word.encode().isalpha() or "sp_" in word or "tm_" in word
            or "runners-up" in word) and (word.upper() != word):
            idx = i
            break
    subj = " ".join(tlist[0:idx]).strip()
    rel = tlist[idx]
    obj = " ".join(tlist[idx+1:]).strip()
    return [subj, rel, obj]


def normalization(score):
    if score < 0:
        score = 0
    return score * 5


def threshold(t=0.5):
    mapping = Mapping()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta")
    model_config = {
        "model_path": "BM-K/KoSimCSE-roberta",
        "cos": nn.CosineSimilarity(),
        "device": device
    }
    model = SimilarityScorer(model_config).to(device)
    model.eval()

    def tokenize(q):
        if not q:
            return None
        return tok(q, add_special_tokens=True, max_length=100,
                   padding='max_length', return_tensors="pt", truncation=True).to(device)

    filenames = ["train", "valid", "test"]
    for filename in filenames:
        base = pd.read_csv(f"./data-all/{filename}_nli.tsv", sep='\t', encoding="utf-8",
                           names=["query", "pos", "hard", "neg", "sim"])
        base.drop(columns=["hard"], inplace=True)
        base = base.values.tolist()

        outfile = open(f"./data/{filename}.tsv", "w", encoding="utf-8")

        for q, p, n, s in tqdm(base):
            if float(s) < t:
                continue
            if filename == "train":
                outfile.write(f"{q}\t{p}\t{n}\n")
            else:
                nl = list_to_triple(n.split(" "))
                nl = mapping.mapping_to_nl(nl)
                ns = model(tokenize(q), tokenize(nl)).item()

                s = normalization(s)
                ns = normalization(ns)

                outfile.write("%s\t%s\t%.3lf\n" % (q, p, s))
                outfile.write("%s\t%s\t%.3lf\n" % (q, n, ns))
                # outfile.write(f"{q}\t{p}\t{s}\n")
                # outfile.write(f"{q}\t{n}\t{ns}\n")

        outfile.close()


def convert_word(oldword, newword):
    filenames = ["train", "valid", "test"]
    for filename in filenames:
        if filename == "train":
            base = pd.read_csv(f"./data/{filename}.tsv", sep='\t', encoding="utf-8",
                               names=["query", "positive", "negative"])
            base = base.values.tolist()

            with open(f"./data/{filename}.tsv", "w", encoding="utf-8") as fp:
                for q, p, n in tqdm(base):
                    p = re.sub(oldword, newword, p)
                    n = re.sub(oldword, newword, n)
                    fp.write(f"{q}\t{p}\t{n}\n")

        else:
            base = pd.read_csv(f"./data/{filename}.tsv", sep='\t', encoding="utf-8",
                               names=["query", "triple", "sim"])
            base = base.values.tolist()

            with open(f"./data/{filename}.tsv", "w", encoding="utf-8") as fp:
                for q, t, s in tqdm(base):
                    t = re.sub(oldword, newword, t)
                    fp.write(f"{q}\t{t}\t{s}\n")


if __name__ == "_main__":
    threshold()
    # convert_word("까벨", "케이블")
    # convert_word("대아미리가합중국", "미국")
