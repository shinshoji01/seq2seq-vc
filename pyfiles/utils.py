import torch

class Dict2Obj(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])
            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)