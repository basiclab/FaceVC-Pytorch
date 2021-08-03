import os
import argparse
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator, Domain_Trans, FaceEncoder


parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, default=4)
parser.add_argument('--outdir', type=str, default='reb_stage3_nofixGpse_tune1')

# stage I  : fill in G_pse_path
# stage II : fill in G_ref_path
# stage III: fill in G_pse_path, G_ref_path, W_path
parser.add_argument('--G_pse_path', type=str, default='checkpoint/reb_stage3_nofixGpse_tune1/G.ckpt', help='model path')
parser.add_argument('--G_ref_path', type=str, default='pretrain_VC/refG/tune1.ckpt', help='model path')
parser.add_argument('--W_path', type=str, default='', help='model path')
config = parser.parse_args()


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:0'

if config.G_pse_path is not '':
    # G_face = Generator(32,512,512,32).eval().to(device)
    G_face = FaceEncoder(32,512,512,32).eval().to(device)
    face_checkpoint = torch.load(config.G_pse_path)
    G_face.load_state_dict(face_checkpoint)
if config.G_ref_path is not '':
    G_sph = Generator(32,256,512,32).eval().to(device)
    sph_checkpoint = torch.load(config.G_ref_path)
    G_sph.load_state_dict(sph_checkpoint)
if config.W_path is not '':
    Warp = Domain_Trans().eval().to(device)
    warp_checkpoint = torch.load(config.W_path)
    Warp.load_state_dict(warp_checkpoint)

if os.path.exists(os.path.join('wav', config.outdir)) is False:
    os.mkdir(os.path.join('wav', config.outdir))

spect_vc = []
# train
src_speaker_lst = ['p286_001.npy', 'p258_031.npy', 'p266_243.npy', 'p333_027.npy']
tgt_speaker_lst = ['0af00UcTOSc-00001.npy', '0C5UQbWzwg8-00001.npy', '0FQXicAGy5U-00001.npy', '0HEXx3SP8kk-00001.npy', '0ITHlySbhJE-00001.npy',\
                   '0akiEFwtkyA-00001.npy', '0d6iSvF1UmA-00001.npy', '0FkuRwU8HFc-00001.npy', '0JGarsZE1rk-00001.npy', '0N6cjPWqpSk-00001.npy',\
                   '01GWGmg5jn8-00001.npy', '03h0dNZoxr8-00001.npy', '05jJodDVJRQ-00002.npy', '06M8qY7Q74Y-00001.npy', '08ZWROqoTZo-00001.npy',\
                   '0SW0HFy9Et4-00001.npy', '0wpCZxiAQzw-00001.npy', '0ZhL7P7w3as-00001.npy', '1bXAkbCyjpo-00001.npy', '1BXYSGepx7Q-00001.npy']

for i, src_speaker in enumerate(src_speaker_lst):
    src_speaker_mel = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spmel16', src_speaker.split('_')[0], src_speaker))
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        src_speaker_mel = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spmel16', src_speaker.split('_')[0], src_speaker))
        src_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spk_emb16', src_speaker.split('_')[0]+'.npy'))
        tgt_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/lrs3/faceemb_512_mtcnn_margin50', tgt_speaker))

        src_speaker_mel, len_pad = pad_seq(src_speaker_mel)
        src_speaker_mel = torch.from_numpy(src_speaker_mel[np.newaxis, :, :]).to(device)
        src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).to(device)
        
        if config.stage == 1:
            with torch.no_grad():
                _, x_identic_psnt, _, _ = G_face(src_speaker_mel, src_speaker_emb, tgt_speaker_emb, None, None)
        if config.stage == 2:
            with torch.no_grad():
                _, x_identic_psnt, _, _ = G_sph(src_speaker_mel, src_speaker_emb, tgt_speaker_emb, None, None)
        if config.stage == 3:
            with torch.no_grad():
                tgt_speaker_emb = G_face(None, None, tgt_speaker_emb, None, None)
                tgt_speaker_emb = Warp(tgt_speaker_emb)
                _, x_identic_psnt, _, _ = G_sph(src_speaker_mel, src_speaker_emb, tgt_speaker_emb, None, None)
                
        if len_pad == 0:
            uttr_trg = x_identic_psnt.unsqueeze(0)[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt.unsqueeze(0)[0, 0, :-len_pad, :].cpu().numpy()
            
        print(uttr_trg.shape)
        spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))
        
    

        
with open(os.path.join('wav', config.outdir, 'results.pkl'), 'wb') as handle:
    pickle.dump(spect_vc, handle)