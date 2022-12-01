import os
import torch
import numpy as np


def sentence_to_triple(sentence):
    sentence = sentence.split(" ")
    for i, word in enumerate(sentence):
        if (word.encode().isalpha() or "sp_" in word or "tm_" in word
            or "runners-up" in word) and (word.upper() != word):
            idx = i
            break
    subj = " ".join(sentence[0:idx]).strip()
    rel = sentence[idx]
    obj = " ".join(sentence[idx+1:]).strip()
    return [subj, rel, obj]


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_log():
    log = {
        "train_loss": 0.0,
        "valid_loss": 0.0,
        "valid_acc": 0.0,
        "min_loss": None,
        "max_spear": 0.0,
        "spear": 0.0,
        "pearson": 0.0
    }
    return log


def cal_accuracy(a):
    collect = 0
    for i in range(len(a)):
        if torch.topk(a[i], 1).indices == i:
            collect += 1
    return collect / a.shape[0]


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt / 0.05


def normalization(score):
    if score < 0:
        score = 0
    return score * 5
