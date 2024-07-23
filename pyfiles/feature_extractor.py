import torch
import librosa

import sys
sys.path.append("./../../unilm/wavlm/")
from WavLM import WavLM, WavLMConfig

# Vocos
sys.path.append("/mntcephfs/lab_data/shoinoue/Models/trained_models/vocos/vocos16k_noncausal_tealab/")
from vocos16k_inference import Vocos
from vocos.feature_extractors import MelSpectrogramFeatures, EncodecFeatures

class WavLMExtractor():
    def __init__(self, fs=16000, ckpt_path="/mntcephfs/lab_data/shoinoue/Models/trained_models/wavlm/WavLM-Large.pt", device="cuda"):
        self.fs = fs
        checkpoint = torch.load(ckpt_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval();
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
    
    def get_feature(self, path):
        wav, _ = librosa.load(path, self.fs)
        wav_input_16khz = torch.tensor(wav).unsqueeze(0)
        if self.cfg.normalize:
            wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
        rep = self.model.extract_features(wav_input_16khz.to(self.device))[0][0]
        return rep
    
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