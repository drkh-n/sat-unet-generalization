import yaml
import os 


def init_ckpt_folder(cfg_name, cfg):
    cfg_v = cfg['xp_config']['cfg_version']
    ckpt_folder = os.path.join('checkpoints', cfg_name, f"{cfg_v}")
    os.makedirs(ckpt_folder, exist_ok=True)
    with open(os.path.join(ckpt_folder, 'hparams.yml'), 'w') as file:
        yaml.dump(cfg, file)
    return ckpt_folder 