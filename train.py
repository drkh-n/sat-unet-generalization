import argparse
from pathlib import Path
import yaml

from utils import logs

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from utils import init_ckpt_folder, ResumedEarlyStopping
from models import get_model
from data import CloudDataModule

import trainer as t
import trainer_multi_gpu as tmg

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


def main(hparams_file):
    # Loading config file
    cfg_name = Path(hparams_file).stem
    cfg = yaml.load(open(hparams_file), Loader=yaml.FullLoader)

    # Securely load Comet API key from environment variable
    if cfg.get('logs', {}).get('type') == 'comet':
        comet_api_key = os.getenv('COMET_API_KEY')
        if comet_api_key:
            cfg['logs']['comet']['api_key'] = comet_api_key

    # Init ckpt folder
    ckpt_folder = init_ckpt_folder(cfg_name, cfg)

    mode = cfg['training']['type']
    # Seed
    if mode == "lightning":
        pl.seed_everything(42)
        # Callbacks
        early_stop_callback = (ResumedEarlyStopping if cfg['resumed'] else EarlyStopping)(monitor='valid_loss', **cfg['early_stop'])
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder, **cfg['model_ckpt'])
    else:
        torch.manual_seed(42)
    # Load data 
    datamodule = CloudDataModule(**cfg['data'])
    # Load model
    model_class = get_model(cfg['xp_config']['model_name'])
    model = model_class(**cfg['model'])
    # Log
    if mode != "multi_gpu":
        logger = logs.logger(cfg['logs'], cfg_name)

    if mode == "lightning":
        print("Training using PyTorch Lightning")
    # Trainer
        trainer = pl.Trainer(**cfg['trainer'],
                            logger=logger,
                            enable_checkpointing=checkpoint_callback,
                            callbacks=[checkpoint_callback, early_stop_callback])
        # Train
        trainer.fit(model=model, 
                    datamodule=datamodule, 
                    ckpt_path=os.path.join(ckpt_folder, 'last.ckpt') if os.path.exists(os.path.join(ckpt_folder, 'last.ckpt')) else None)

    elif mode == "torch":
        print("Training using PyTorch")
        trainer = t.Trainer(cfg['trainer'],
                        logger=logger)
    
        trainer.fit(model=model, 
                    datamodule=datamodule, 
                    ckpt_path=ckpt_folder,
                    early_stop=cfg['early_stop'])

    elif mode == "multi_gpu":
        print("Training using PyTorch with multiple gpus")
        trainer = tmg.Trainer(cfg['trainer'],
                    logger=cfg['logs'])
    
        trainer.fit(model=model, 
                    datamodule=datamodule, 
                    ckpt_path=ckpt_folder,
                    early_stop=cfg['early_stop'],
                    gpus=cfg["training"]["multi_gpu"]
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, default="./configs/satunet.yml", help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)