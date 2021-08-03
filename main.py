import os
import argparse
from solver_encoder import Solver
from data_loader_clean import get_loader_clean
from data_loader_noisy import get_loader_noisy
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader
    vcc_loader_clean = get_loader_clean(config.batch_size, config.len_crop)
    vcc_loader_noisy = get_loader_noisy(config.batch_size, config.len_crop)
    
    if config.stage == 1 or config.stage == 3:
        solver = Solver(vcc_loader_noisy, config)
    elif config.stage == 2:
        solver = Solver(vcc_loader_clean, config)

    solver.train()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--stage', type=int, default=1)

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
    parser.add_argument('--refG_path', type=str, default='pretrain_VC/refG/tune1.ckpt', help='refG model name')

    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=2000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    parser.add_argument('--clip', type=int, default=1, help='clip value of gradient clip')
    parser.add_argument('--model_id', type=str, default='reb_stage3_nofixGpse_tune1', help='model name')
    
    # Logging and checkpointing.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_step', type=int, default=1000)
    
    # Data configuration.

    config = parser.parse_args()
    print(config)
    main(config)