# FaceVC

This is the official implementation for "Face-based Voice Conversion: Learning the Voice behind a Face" (FaceVC).

The audio demo for FaceVC can be found at https://facevc.github.io/.

### Data Preprocessing

For making face embedding, please refer to https://github.com/timesler/facenet-pytorch.

For making speaker embedding and spectrogram, please refer to https://github.com/auspicious3000/autovc.

#### In-the-wild data
1. Prepare a data list of all the training utterance path (for making speaker dictionary).
2. Prepare face embedding / speaker embedding / spectrogram of in-the-wild data.
3. Set following path in ```data_loader_noisy.py```.
```python
spk_lst = ''
root_face = ''
root_speech = ''
root_mel = ''
```
#### Lab-collected data
1. Prepare a data list of all the training utterance path (for making speaker dictionary).
2. Prepare speaker embedding / spectrogram of lab-collected data.
3. Set following path in ```data_loader_clean.py```.
```python
spk_lst = ''
root_speaker = ''
root_mel = ''
```


### Training
1. Create environment.
```bash
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```
2. Set configuration in ```main.py``` according to the training stage.
```python
    parser.add_argument('--stage', type=int, default=3)

    # Model configuration.
    ### Generator for stage I or pseudo generator for stage III ###
    ### Note: Weight of reconstruction loss is set to 1. ###
    parser.add_argument('--lambda_cd_pse', type=float, default=0.1, help='weight for hidden code loss')#1
    parser.add_argument('--lambda_ge2e_pse', type=float, default=0.05, help='weight for ge2e loss')
    parser.add_argument('--dim_neck_pse', type=int, default=32)
    parser.add_argument('--dim_emb_pse', type=int, default=512)
    parser.add_argument('--dim_pre_pse', type=int, default=512)
    parser.add_argument('--freq_pse', type=int, default=32)

    ### Generator for stage II or referance generator for stage III ###
    ### Note: Weight of reconstruction loss is set to 1. ###
    parser.add_argument('--lambda_cd_ref', type=float, default=0.1, help='weight for hidden code loss')#1
    parser.add_argument('--dim_neck_ref', type=int, default=32)
    parser.add_argument('--dim_emb_ref', type=int, default=256)
    parser.add_argument('--dim_pre_ref', type=int, default=512)
    parser.add_argument('--freq_ref', type=int, default=32)

    # Training configuration.
    ### Loading pretrained pseudo generator (from stage I) / referance generator (from stage II) for stage III ###
    parser.add_argument('--pseG_path', type=str, default='pretrain_VC/pseG/G.ckpt', help='pseG model name')
    parser.add_argument('--refG_path', type=str, default='pretrain_VC/refG/G.ckpt', help='refG model name')

    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=2000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    parser.add_argument('--clip', type=int, default=1, help='clip value of gradient clip')
    parser.add_argument('--model_id', type=str, default='test', help='model name')
    
    # Logging and checkpointing.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_step', type=int, default=1000)
```

3. Run main.py
```bash
$ python main.py
```

4. Tensorboard
```
$ tensorboard --logdir log --host tunnel_host --port tunnel_port
```

### Testing
1. Set configuration in ```test_conversion.py``` according to the training stage.
```python
parser.add_argument('--stage', type=int, default=4)
parser.add_argument('--outdir', type=str, default='reb_stage3_nofixGpse_tune1')

# stage I  : fill in G_pse_path
# stage II : fill in G_ref_path
# stage III: fill in G_pse_path, G_ref_path, W_path
parser.add_argument('--G_pse_path', type=str, default='', help='model path')
parser.add_argument('--G_ref_path', type=str, default='', help='model path')
parser.add_argument('--W_path', type=str, default='', help='model path')
```

2. Run test_conversion.py
```
$ python test_conversion.py
```

3. vocoder.py
```
$ python vocoder.py
```
