import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef

from config import get_train_config
from argument import get_args
from utils import seed_everything, get_train_log, normalization, sim_matrix

writer = SummaryWriter()

def train(args):
    config = get_train_config(args)
    log = get_train_log()

    print("Training Start")
    for e in range(config["epoch"]):
        print("Epoch {}".format(e))
        log["train_loss"] = 0.0
        log["valid_loss"] = 0.0
        log["valid_acc"] = 0.0

        batch_id = 0
        config["model"].train()
        for i, (query, positive, _) in tqdm(enumerate(config["train_loader"]), desc="[TRAIN]",
                                                   total=len(config["train_loader"])):
            label = torch.tensor([n for n in range(len(query))], dtype=torch.long, device="cuda")

            q_embed = config["model"](query)
            p_embed = config["model"](positive)

            q_sim = sim_matrix(q_embed, p_embed)

            config["optim"].zero_grad()
            loss = config["loss"](q_sim, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(config["model"].parameters(), config["grad_clip"])
            config["optim"].step()
            config["scheduler"].step()

            log["train_loss"] += loss.item()
            writer.add_scalar(f"loss/train", log["train_loss"], e)
            batch_id += 1

            if i != 0 and i % config["save_interval"] == 0:
                _validation(config, log)

        log["train_loss"] /= batch_id

        _validation(config, log)
        print("[TRAIN] training Loss {}\n".format(log["train_loss"]))
    writer.close()


def _validation(config, log):
    batch_id = 0

    spearman = 0.0
    pearson = 0.0

    config["model"].eval()
    for i, (query, triple, tri_score) in tqdm(enumerate(config["valid_loader"]), desc="[VALID]",
                                              total=len(config["valid_loader"])):
        tri_score = tri_score.squeeze()

        q_embed = config["model"](query)
        t_embed = config["model"](triple)

        t_sim = config["cos"](q_embed.cpu(), t_embed.cpu())
        t_sim = torch.tensor([round(normalization(t), 3) for t in t_sim.tolist()], dtype=torch.float64)

        spear = spearman_corrcoef(t_sim, tri_score)
        pear = pearson_corrcoef(t_sim, tri_score)

        spearman += spear.item()
        pearson += pear.item()

        del spear
        del pear
        torch.cuda.empty_cache()

        batch_id += 1
    spearman /= batch_id
    pearson /= batch_id

    print(f"\n[VALID] Spearman\t{spearman}")
    print(f"[VALID] Pearson\t{pearson}\n")

    if log["max_spear"] < spearman:
        print("Spearman {} ------> {}\n".format(log["max_spear"], spearman))
        log["max_spear"] = spearman

        if not os.path.exists(config["outpath"]):
            os.mkdir(config["outpath"])
        torch.save(config["model"].state_dict(),
                   os.path.join(config["outpath"], config["model_name"] + ".pth"))


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train(args)
