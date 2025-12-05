from glob import glob 
import os
import numpy as np
from tqdm import tqdm
import gc
import re
import argparse
import csv
from pathlib import Path
import yaml

from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall

from data import CloudDataModule
from models import get_model
from inference import Segmentator
import torch
import gc
from tqdm import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(f"Using device: {device}")

def main(hparams, checkpoint):

    bn_auroc = AUROC(task="binary")
    bn_acc   = Accuracy(task="binary")
    bn_f1    = F1Score(task="binary")
    bn_prec  = Precision(task="binary")
    bn_rec   = Recall(task="binary")


    results = []

    # model_ckpt_path = "/mnt/d/checkpoints/38/model_185_0.9624.pt"
    model_cfg_path = hparams
    model_ckpt_path = checkpoint

    print(f"{model_cfg_path}")

    with open(model_cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    datamodule = CloudDataModule(**cfg["data"])
    print(cfg["data"]["data_root"])

    datamodule.setup(stage="fit")
    test_dataloader = datamodule.val_dataloader()

    predictor = Segmentator(model_cfg_path, model_ckpt_path)

    for inputs, labels in tqdm(test_dataloader):
        inputs, targets = inputs, labels
        with torch.no_grad():
            probs = predictor.predict(inputs, threshold=None)
        probs = torch.from_numpy(probs)
        # targets = torch.from_numpy(targets)
        bn_auroc.update(preds=probs, target=targets)
        bn_acc.update(preds=probs, target=targets)
        bn_f1.update(preds=probs, target=targets)
        bn_prec.update(preds=probs, target=targets)
        bn_rec.update(preds=probs, target=targets)

    auc_value = bn_auroc.compute().item()
    acc_value = bn_acc.compute().item()
    f1_value = bn_f1.compute().item()
    prec_value = bn_prec.compute().item()
    rec_value = bn_rec.compute().item()

    print(f"Result -> Model: {model_cfg_path} | CKPT: {model_ckpt_path} "
    f"-> AUROC: {auc_value:.4f} | ACC: {acc_value:.4f} | F1: {f1_value:.4f} | PREC: {prec_value:.4f} | REC: {rec_value:.4f}")

    results.append({
        "model": model_cfg_path,
        "checkpoint": model_ckpt_path,
        "auc": auc_value,
        "acc": acc_value,
        "f1": f1_value,
        "precision": prec_value,
        "recall": rec_value
    })
    print('end line')
    bn_auroc.reset()
    bn_acc.reset()
    bn_f1.reset()
    bn_prec.reset()
    bn_rec.reset()
    del datamodule
    del predictor
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, default="./configs/satunet.yml", help="hparams config file")
    parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoints/satunet-aug-norm/torch4/last.ckpt")
    args = parser.parse_args()
    main(args.hparams, args.checkpoint)