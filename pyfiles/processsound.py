import torch
import librosa
# import torchaudio
from scipy.io.wavfile import write as write_wav

def trim_audio_and_save(path, fs=16000, savepath="temp.wav", trim_threshold_in_db=30, trim_frame_size=2048, trim_hop_size=512):
    audio, _ = librosa.load(path, sr=fs)
    audio, _ = librosa.effects.trim(
        audio,
        top_db=trim_threshold_in_db,
        frame_length=trim_frame_size,
        hop_length=trim_hop_size,
    )
    # torchaudio.save(savepath, torch.tensor(audio).unsqueeze(0), fs)
    write_wav(savepath, fs, audio)
    return 
