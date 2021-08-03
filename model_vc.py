import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, dim_emb_in, dim_emb_out):
        super(Self_Attn,self).__init__()
        
        self.query = LinearNorm(dim_emb_in, dim_emb_out)
        self.key = LinearNorm(dim_emb_in, dim_emb_out)
        self.value = LinearNorm(dim_emb_in, dim_emb_out)

        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( dim_emb )
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        x = x.unsqueeze(1)
        proj_query  = self.query(x)
        proj_key =  self.key(x)
        energy =  torch.matmul(proj_query.permute(0,2,1), proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x)

        out = torch.matmul(proj_value, attention.permute(0,2,1))
        out = out.squeeze(1)
        
        return out, attention

class Repara(nn.Module):

    def __init__(self, dim_emb, hidden_sizes, latent_size):
        super(Repara,self).__init__()

        self.linearelu = nn.Sequential(
                nn.Linear(dim_emb, hidden_sizes),
                nn.ReLU(),
                nn.Linear(hidden_sizes, latent_size),
                nn.ReLU())

        self.linearelu_mu = nn.Linear(latent_size, latent_size)
        self.linearelu_logvar = nn.Linear(latent_size, latent_size)


    def gaussian_param_projection(self, x):
        return self.linearelu_mu(x), self.linearelu_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.linearelu(x)
        mu, logvar = self.gaussian_param_projection(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
        
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
            
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.attn = Self_Attn(dim_emb, dim_emb)
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, 256, dim_pre)
        self.postnet = Postnet()
        self.reparam = Repara(dim_emb, hidden_sizes = 320, latent_size = 256)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x, c_org, c_trg, ge2e_pack): 
    # ge2e_pack: spk_num * emb_per_spk * emb_size; domain_input: None in conversion phase; alpha in DAT
        
        if x is None:
            return c_trg
       
        if ge2e_pack is not None:
            ge2e_ip = []
            for emb in ge2e_pack:
                emb, attn_map = self.attn(emb)
                emb_V2A, _, _ = self.reparam(emb)
                ge2e_ip.append(emb_V2A)
            return torch.stack(ge2e_ip, dim=0)
    

        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
            
        c_trg, attn_map = self.attn(c_trg)
        c_trg, mu, logvar = self.reparam(c_trg)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1), c_trg


class Domain_Trans(nn.Module):
    """Domain_Transformation network."""
    def __init__(self):
        super(Domain_Trans, self).__init__()
        
        self.trans_layer = nn.Linear(256, 256)
        
    def forward(self, feat): 
        feat = self.trans_layer(feat)
        return feat

class FaceEncoder(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(FaceEncoder, self).__init__()
        
        self.attn = Self_Attn(dim_emb, dim_emb)
        self.reparam = Repara(dim_emb, hidden_sizes = 320, latent_size = 256)
        self.trans_layer = nn.Linear(256, 256)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, c_trg): 

        c_trg, attn_map = self.attn(c_trg)
        c_trg, mu, logvar = self.reparam(c_trg)
        c_trg = self.trans_layer(c_trg)
        
        
        return c_trg