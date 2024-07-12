import torch
import librosa

import sys
sys.path.append("./../../unilm/wavlm/")
from WavLM import WavLM, WavLMConfig

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