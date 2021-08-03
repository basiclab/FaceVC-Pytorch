from model_vc import Generator, Domain_Trans, FaceEncoder
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import time
import os
import datetime
from tensorboardX import SummaryWriter 
from ge2e import GE2ELoss


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # stage
        self.stage = config.stage
        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd_pse = config.lambda_cd_pse
        self.lambda_ge2e_pse = config.lambda_ge2e_pse
        self.dim_neck_pse = config.dim_neck_pse
        self.dim_emb_pse = config.dim_emb_pse
        self.dim_pre_pse = config.dim_pre_pse
        self.freq_pse = config.freq_pse

        self.lambda_cd_ref = config.lambda_cd_ref
        self.dim_neck_ref = config.dim_neck_ref
        self.dim_emb_ref = config.dim_emb_ref
        self.dim_pre_ref = config.dim_pre_ref
        self.freq_ref = config.freq_ref

        # Training configurations.
        self.pseG_path = config.pseG_path
        self.refG_path = config.refG_path
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.clip = config.clip
        self.model_id = config.model_id
        
        # Logging and checkpointing.
        self.use_cuda = torch.cuda.is_available()
        print("use_cuda: "+ str(self.use_cuda))
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.writer = SummaryWriter(os.path.join('log', self.model_id))
        print('Save log in: '+os.path.join('log', self.model_id))
        self.criterion_ge2e = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax').to(self.device) #for softmax loss
        
        # Build the model and tensorboard.
        self.build_model()

            
    def build_model(self):

        if self.stage == 1:
            self.G_pse = Generator(self.dim_neck_pse, self.dim_emb_pse, self.dim_pre_pse, self.freq_pse).to(self.device)
            self.g_optimizer = torch.optim.Adam(self.G_pse.parameters(), 0.0001)
        elif self.stage == 2:
            self.G_ref = Generator(self.dim_neck_ref, self.dim_emb_ref, self.dim_pre_ref, self.freq_ref).to(self.device)
            self.g_optimizer = torch.optim.Adam(self.G_ref.parameters(), 0.0001)
        elif self.stage == 3:
            g_pse_checkpoint = torch.load(self.pseG_path)
            g_ref_checkpoint = torch.load(self.refG_path)
            self.G_pse = Generator(self.dim_neck_pse, self.dim_emb_pse, self.dim_pre_pse, self.freq_pse).eval().to(self.device)
            self.G_ref = Generator(self.dim_neck_ref, self.dim_emb_ref, self.dim_pre_ref, self.freq_ref).eval().to(self.device)
            self.G_pse.load_state_dict(g_pse_checkpoint)
            self.G_ref.load_state_dict(g_ref_checkpoint)
        
            self.Warp = Domain_Trans().to(self.device)
            self.w_optimizer = torch.optim.Adam(self.Warp.parameters(), 0.0001)
        
        
    def reset_grad(self):
        """Reset the gradient buffers."""
        if self.stage == 1 or self.stage == 2:
            self.g_optimizer.zero_grad()
        elif self.stage == 3:
            self.w_optimizer.zero_grad()
        elif self.stage == 4:
            self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        if self.stage == 1:
            keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd','G/loss_ge2e']
        elif self.stage == 2:
            keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
        elif self.stage == 3:
            keys = ['W/loss_warping']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            if self.stage == 1 or self.stage == 3:
                try:
                    x_real, emb_org, ge2e_pack, emb_sph = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, emb_org, ge2e_pack, emb_sph = next(data_iter)
                x_real = x_real.to(self.device)
                emb_org = emb_org.to(self.device)
                ge2e_pack = ge2e_pack.to(self.device)
                emb_sph = emb_sph.to(self.device)
            elif self.stage == 2:
                try:
                    x_real, emb_org = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, emb_org = next(data_iter)
                x_real = x_real.to(self.device)
                emb_org = emb_org.to(self.device)

           
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            with torch.autograd.set_detect_anomaly(True):
                if self.stage == 1:
                    self.G_pse = self.G_pse.train()
                elif self.stage == 2:
                    self.G_ref = self.G_ref.train()
                elif self.stage == 3:
                    self.Warp = self.Warp.train()

                # =================================================================================== #
                #                               2-1. Train G                                          #
                # =================================================================================== #
                
                if self.stage == 1:
                    # Identity mapping loss
                    x_identic, x_identic_psnt, code_real, _ = self.G_pse(x_real, emb_org, emb_org, None)
                    g_loss_id = F.l1_loss(x_real, x_identic)
                    g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)
                    
                    # Code semantic loss.
                    code_reconst = self.G_pse(x_identic_psnt, emb_org, None, None)
                    g_loss_cd = F.l1_loss(code_real, code_reconst)
                    
                    # KL loss
                    # g_loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    # ge2e loss
                    ge2e_ip = self.G_pse(None, None, None, ge2e_pack)
                    g_loss_ge2e = self.criterion(ge2e_ip) 

                    # Backward and optimize.
                    g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd + self.lambda_ge2e * g_loss_ge2e
                    self.reset_grad()
                    g_loss.backward()
                    clip_grad_norm_(filter(lambda p: p.requires_grad, self.G_pse.parameters()), self.clip)
                    self.g_optimizer.step()

                    # Logging.
                    loss = {}
                    loss['G/loss_id'] = g_loss_id.item()
                    loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
                    loss['G/loss_cd'] = g_loss_cd.item()
                    loss['G/loss_ge2e'] = g_loss_ge2e.item()
                    self.writer.add_scalar('G/loss_id', g_loss_id.item(), i+1)
                    self.writer.add_scalar('G/loss_id_psnt', g_loss_id_psnt.item(), i+1)
                    self.writer.add_scalar('G/loss_cd', g_loss_cd.item(), i+1)
                    self.writer.add_scalar('G/loss_ge2e', g_loss_ge2e.item(), i+1)
                
                elif self.stage == 2:
                    # Identity mapping loss
                    x_identic, x_identic_psnt, code_real, _ = self.G_ref(x_real, emb_org, emb_org, None)
                    g_loss_id = F.l1_loss(x_real, x_identic)
                    g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)
                    
                    # Code semantic loss.
                    code_reconst = self.G(x_identic_psnt, emb_org, None, None)
                    g_loss_cd = F.l1_loss(code_real, code_reconst)
                    
                    # KL loss
                    # g_loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    # ge2e loss
                    # g_loss_ge2e = self.criterion(ge2e_ip) 

                    # Backward and optimize.
                    g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss = {}
                    loss['G/loss_id'] = g_loss_id.item()
                    loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
                    loss['G/loss_cd'] = g_loss_cd.item()
                    self.writer.add_scalar('G/loss_id', g_loss_id.item(), i+1)
                    self.writer.add_scalar('G/loss_id_psnt', g_loss_id_psnt.item(), i+1)
                    self.writer.add_scalar('G/loss_cd', g_loss_cd.item(), i+1)

                elif self.stage == 3:
                    sph_spk = self.G_ref(None, None, emb_sph, None)
                    face_spk = self.G_pse(None, None, emb_org, None)
                    face_spk = self.Warp(face_spk)
                    
                    w_loss = F.mse_loss(sph_spk, face_spk)
                
                    self.reset_grad()
                    w_loss.backward()
                    self.w_optimizer.step()
                    
                    loss = {}
                    loss['W/loss_warping'] = w_loss.item()
                    self.writer.add_scalar('W/loss_warping', w_loss.item(), i+1)
                    
                
                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training information.
                if i == 0:
                    if not os.path.exists(os.path.join('checkpoint', self.model_id)):
                        os.mkdir(os.path.join('checkpoint', self.model_id))
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                    for tag in keys:
                        log += ", {}: {:.4f}".format(tag, loss[tag])
                    print(log)
                if (i+1) % self.save_step == 0:
                    if self.stage == 1:
                        torch.save(self.G_pse.state_dict(), os.path.join('checkpoint', self.model_id, 'G.ckpt'))
                        torch.save(self.g_optimizer.state_dict(), os.path.join('checkpoint', self.model_id, 'op_g.ckpt'))
                    elif self.stage == 2:
                        torch.save(self.G_ref.state_dict(), os.path.join('checkpoint', self.model_id, 'G.ckpt'))
                        torch.save(self.g_optimizer.state_dict(), os.path.join('checkpoint', self.model_id, 'op_g.ckpt'))
                    elif self.stage == 3:
                        torch.save(self.Warp.state_dict(), os.path.join('checkpoint', self.model_id, 'W.ckpt'))
                    
                    print('Save ckpt in: '+os.path.join('checkpoint', self.model_id))
                
