import torch

import sys
sys.path.append("/mntcephfs/lab_data/shoinoue/Models/trained_models/vocos/vocos16k_noncausal_tealab/")
from vocos16k_inference import Vocos
from vocos.feature_extractors import MelSpectrogramFeatures, EncodecFeatures

def get_vocos(config_path, model_path, fs):
    if fs==24000:
        model = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(model.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in model.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        model.load_state_dict(state_dict)
        model.eval();
    elif fs==16000:
        model = Vocos.from_hparams(config_path)
        state_dict_list = torch.load(model_path, map_location="cpu")
        state_dict = state_dict_list['state_dict']
        model_state_dict = model.state_dict()
        new_state_dict = {}
        for key, v in state_dict.items():
            if key in model_state_dict.keys():
                # print(key)
                new_state_dict[key] = v

        if isinstance(model.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in model.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        model.load_state_dict(new_state_dict, strict=True)
        model.eval();
    return model


class Dict2Obj(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])
            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)