# from .transunet import TransUNet
from .satunet import SatUNet

MODELS = {"SatUNet": SatUNet}

def get_model(model_name):
    valid_model_names = [
    'SatUNet'
    ]
    assert model_name in valid_model_names, 'Invalid model name'
    return MODELS[model_name]