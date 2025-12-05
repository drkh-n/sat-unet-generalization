from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from comet_ml import Experiment
import torch.distributed as dist
    
def logger(cfg, cfg_name):
    if cfg['type'] == 'tb_logs':     
        return TensorBoardLogger('tb_logs', name=cfg_name, **cfg['tb_logs'])
    elif cfg['type'] == 'comet_light':  
        logger = CometLogger(**cfg['comet'])
        logger.log_hyperparams({"batch_size": cfg['batch_size']})
        return logger
    elif cfg['type'] == 'comet':  
        return Experiment(**cfg['comet'])
    #if cfg_name == 'comet_multi_gpu':  
    #    return Experiment(**cfg['comet'], disabled=(dist.get_rank()!=0))
    
    