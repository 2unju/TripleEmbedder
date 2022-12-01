import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AdamW
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from dataloader import TEDataset, TEVDataset
from model import KoSimCT


def get_train_config(args):
    model_confg = {
        "model_path": args.model_path,
        "device": args.device
    }
    model = KoSimCT(model_confg)
    # model = nn.DataParallel(model, device_ids=list(range(2)))
    # torch.multiprocessing.set_start_method('spawn')
    model.to(args.device)

    tok = AutoTokenizer.from_pretrained(args.model_path)

    train_data_config = {
        "tokenizer": tok,
        "data_path": args.train_path,
        "max_len": args.max_len,
        "device": args.device
    }
    valid_data_config = {
        "tokenizer": tok,
        "data_path": args.valid_path,
        "max_len": args.max_len,
        "device": args.device
    }

    trainset = TEDataset(train_data_config)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(args.batch_size))
    validset = TEVDataset(valid_data_config)
    validloader = torch.utils.data.DataLoader(validset, batch_size=int(args.batch_size))

    criterion = nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=float(args.lr))

    data_len = len(trainloader)
    num_train_steps = int(data_len / args.batch_size * args.epoch)
    num_warmup_steps = int(num_train_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)

    config = {
        "model": model.to(args.device),
        "train_loader": trainloader,
        "valid_loader": validloader,
        "loss": criterion,
        "optim": optimizer,
        "scheduler": scheduler,
        "epoch": args.epoch,
        "device": args.device,
        "max_grad_norm": args.max_grad_norm,
        "cos": nn.CosineSimilarity(),
        "outpath": args.outpath,
        "grad_clip": args.grad_clip,
        "save_interval": args.save_interval,
        "model_name": args.model_name
    }
    return config


def get_test_config(args):
    model_confg = {
        "model_path": args.model_path,
        "device": args.device
    }
    model = KoSimCT(model_confg)
    if args.fine_tuned == "y":
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ckpt", args.model_name + ".pth")))
    model.to(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    test_data_config = {
        "tokenizer": tok,
        "data_path": args.test_path,
        "max_len": args.max_len,
        "device": args.device
    }

    testset = TEVDataset(test_data_config)
    criterion = nn.CrossEntropyLoss()

    config = {
        "model": model.to(args.device),
        "test_loader": torch.utils.data.DataLoader(testset, batch_size=int(args.batch_size)),
        "loss": criterion,
        "cos": nn.CosineSimilarity(),
        "device": args.device,
        "model_name": args.model_name
    }
    return config

