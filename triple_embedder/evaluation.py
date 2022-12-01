import os
import re

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm

from config import get_test_config
from argument import get_test_args
from dataloader import TEVDataset
from utils import sim_matrix, normalization
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef


def inference(config):
    # model, tokenizer init
    config["model"].eval()

    spearman = 0.0
    pearson = 0.0

    origin = open(f"./output/origin.txt", "w", encoding="utf-8")
    with open(f"./output/{config['model_name']}.txt", "w", encoding="utf-8") as fp:
        for i, (query, triple, tri_score) in tqdm(enumerate(config["test_loader"]), desc="[VALID]",
                                                  total=len(config["test_loader"])):
            tri_score = tri_score.squeeze().to("cuda")

            q_embed = config["model"](query)
            t_embed = config["model"](triple)

            t_sim = config["cos"](q_embed, t_embed)
            t_sim = torch.tensor([round(normalization(t), 3) for t in t_sim.tolist()], dtype=torch.float64,
                                 device="cuda")

            for ori, pred in zip(tri_score, t_sim):
                fp.write(f"{pred.item()}\n")
                origin.write(f"{ori.item()}\n")

            spear = spearman_corrcoef(t_sim.type(torch.float64), tri_score)
            pear = pearson_corrcoef(t_sim.type(torch.float64), tri_score)

            spearman += spear.item()
            pearson += pear.item()

            del spear
            del pear
            torch.cuda.empty_cache()
        origin.close()

    print("\t-------- Result --------\n\n")
    print(f"\tSPEARMAN\t{spearman / len(config['test_loader'])}\n")
    print(f"\tPEARSON\t{pearson / len(config['test_loader'])}\n")
    print("\t------------------------\n\n")


if __name__ == "__main__":
    args = get_test_args()
    config = get_test_config(args)
    inference(config)
