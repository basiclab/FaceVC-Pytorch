# FaceVC

This is the official implementation for "Face-based Voice Conversion: Learning the Voice behind a Face" (FaceVC).

### Data Preprocessing
1. 

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

    ### For training stage I, revise argument of line 39-44, 59-67 ###
    ### For training stage II, revise argument of line 48-52, 59-67 ###
    ### For training stage III, revise argument of line 39-44, 48-52, 56-67 ###

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
tensorboard --logdir log --host tunnel_host --port tunnel_port
```

### Testing
1. Set configuration in ```test_conversion.py``` according to the training stage.
```python
parser.add_argument('--stage', type=int, default=3)
parser.add_argument('--outdir', type=int, default=3)

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
