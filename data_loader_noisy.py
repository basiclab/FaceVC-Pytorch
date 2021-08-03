from torch.utils import data
import torch
import numpy as np
import pickle 
import os    
          
spk_lst = '20sp.txt'
root_face = '/mnt/hdd0/hsiaohan/lrs3/faceemb_512_mtcnn_margin50/'
root_speech = '/mnt/hdd0/hsiaohan/lrs3/spk_emb16/'
root_mel = '/mnt/hdd0/hsiaohan/s2f/mel_clean/'

class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, len_crop):
        """Initialize and preprocess the Utterances dataset."""
        self.len_crop = len_crop
        
        self.data_dict = self.make_data_dict()
        self.speaker_lst = list(self.data_dict.keys())
        
        print('Finished init the dataset...')

            
    ### Make utterance dictionary of each speaker
    def make_data_dict(self):
        with open(spk_lst, 'r') as f:
            lines = f.readlines()
            data = {}
            for line in lines:
                file = line.replace('.mp4\n', '')
                speaker = file.split('-')[0]
                idx = file.split('-')[1]
                
                if speaker not in list(data.keys()):
                    data[speaker] = []
                data[speaker].append(idx)
        return data
        
    def __getitem__(self, index):
        # pick a random speaker
        speaker = self.speaker_lst[index]
        
        # pick random uttr with random crop
        a = np.random.randint(0, len(self.data_dict[speaker]))
        emb_org = np.load(root_face+speaker+'-'+self.data_dict[speaker][a]+'.npy')
        emb_sph = np.load(root_speech+speaker+'.npy')
        tmp = np.load(root_mel+speaker+'-'+self.data_dict[speaker][a]+'.npy')

        if tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - tmp.shape[0]
            uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
        elif tmp.shape[0] > self.len_crop:
            left = np.random.randint(tmp.shape[0]-self.len_crop)
            uttr = tmp[left:left+self.len_crop, :]
        else:
            uttr = tmp
        
        cand = np.random.choice(len(self.data_dict[speaker]), size=5, replace=False)
        ge2e_lst = []
        for spk in cand:
            emb_sp = np.load(root_face+speaker+'-'+self.data_dict[speaker][spk]+'.npy')
            ge2e_lst.append(emb_sp)
            
        return uttr, emb_org, np.array(ge2e_lst), emb_sph
    

    def __len__(self):
        """Return the number of spkrs."""
        return len(self.speaker_lst)
    
    
    

def get_loader_noisy(batch_size=16, len_crop=128, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(len_crop)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader






