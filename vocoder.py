import os
import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen

model_id = "reb_stage3_nofixGpse"
spect_vc = pickle.load(open(os.path.join('wav', model_id, 'results.pkl'), 'rb'))
# spect_vc = pickle.load(open("results.pkl", 'rb'))
device = torch.device("cuda")
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(c.shape)
    waveform = wavegen(model, c=c)   
    librosa.output.write_wav(os.path.join('wav', model_id, name+'.wav'), waveform, sr=16000)