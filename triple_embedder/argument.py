import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outpath", default="./ckpt")

    parser.add_argument("--model-name", choices=["bert", "roberta", "kosim-bert", "kosim-roberta"])
    parser.add_argument("--model-path", default="klue/roberta-base")
    parser.add_argument("--train-path", default="./data/train.tsv")
    parser.add_argument("--valid-path", default="./data/valid.tsv")

    parser.add_argument("--batch-size", default=512)
    parser.add_argument("--max-len", default=32)
    parser.add_argument("--lr", default=0.00005)
    parser.add_argument("--epoch", default=4)
    parser.add_argument("--weight-decay", default=1e-4)
    parser.add_argument("--max-grad-norm", default=1)
    parser.add_argument("--warmup-ratio", default=0.1)
    parser.add_argument("--grad-clip", default=1.0)
    parser.add_argument("--save-interval", default=15, type=int)

    return parser.parse_args()


def get_preprocessing_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda")

    parser.add_argument("--model-name", choices=["bert", "roberta", "kosim-bert", "kosim-roberta",
                                                 "bert-base", "roberta-base"])
    parser.add_argument("--model-path", default="klue/bert-base")
    parser.add_argument("--test-path", default="./data/test.tsv")
    parser.add_argument("--max-len", default=32)
    parser.add_argument("--batch-size", default=128)

    parser.add_argument("--fine-tuned", default="y", choices=["y", "n"])

    return parser.parse_args()
