from .MINet import MINet

def get_model(model):
    return eval(model)
