from .MINet import MINet

# image based method
from .DenseASPP import DenseASPP
from .deeplab import DeepLab
from .pspnet import PSPNet
from .BiSeNet import BiSeNet

def get_model(model):
    return eval(model)
