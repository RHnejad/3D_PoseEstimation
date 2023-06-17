#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from __future__ import absolute_import
# from __future__ import print_function

import torch
import numpy as np
from einops import rearrange

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal(m.weight)

class Linear(torch.nn.Module):
    def __init__(self, linear_size, p_dropout=0.5, BN=True):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p_dropout)
        self.dropout2 = torch.nn.Dropout(p_dropout)

        self.w1 = torch.nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = torch.nn.BatchNorm1d(self.l_size)

        self.w2 = torch.nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = torch.nn.BatchNorm1d(self.l_size)
        
        self.BN = BN

    def forward(self, x):
        y = self.w1(x)
        if self.BN:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        if self.BN:
            y = self.batch_norm2(y)
        y = self.relu2(y)
        y = self.dropout2(y)

        out = x + y

        return out


class LinearModel(torch.nn.Module):
    def __init__(self,
                 i_dim, o_dim,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5,
                 BN=True):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  i_dim #new
        # 3d joints
        self.output_size = o_dim #new

        # process input to linear size
        self.w1 = torch.nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = torch.nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout, BN))
        self.linear_stages = torch.nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = torch.nn.Linear(self.linear_size, self.output_size)

        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(self.p_dropout)

        self.flat =  torch.nn.Flatten() #new
        
        self.BN = BN

    def forward(self, x):
        # pre-processing
        y = self.flat(x) #new
        y = self.w1(y) #new (x)
        if self.BN:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y

#_____MINE_____
class MLP(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=2, n_joints=17 ):
        super().__init__()
         
        self.input_dim = input_dim *n_joints
        self.output_dim = output_dim *n_joints

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),

            torch.nn.Linear(self.input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(32, self.output_dim),          
        )
  
    def forward(self, inp):
        outp = self.encoder2(inp)
        return outp
#______ 

class AE(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=3, n_joints=17 ):
        super().__init__()
         
        self.input_dim = input_dim *n_joints
        self.output_dim = output_dim *n_joints

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),

            torch.nn.Linear(self.input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(16, 12),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
           
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(12, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(128, self.output_dim )         
        )
        self.encoder2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.input_dim, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5) )

        self.decoder2 = torch.nn.Sequential(

            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5) ,
           
            torch.nn.Linear(1024, self.output_dim ) ) 

        
        self.acti_final = torch.nn.Tanh()

    def forward(self, x):
        encoded = self.encoder2(x)
        decoded = self.decoder2(encoded)
        if self.output_dim == 2 :
            decoded = self.acti_final(decoded)
        return decoded
#______ 

#______

def get_positional_embeddings(sequence_length, d=2):
    result = torch.ones(sequence_length,d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i/((1e4)**(j/d))) if j%2==0 else np.cos(i/((1e4)**((j-1)/d)))
    return result


class Attention(torch.nn.Module): #https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = torch.nn.LayerNorm(dim)

        self.attend = torch.nn.Softmax(dim = -1)

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = torch.nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MyMSA(torch.nn.Module): #MultiHeadSelfAttention
    def __init__(self, d=2, n_heads =1):
        super(MyMSA, self).__init__()
        self.d = d 
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimention {d} into {n_heads} heads"

        d_head = int(d/n_heads)
        self.q_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range (self.n_heads)])
        self.k_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range (self.n_heads)])
        self.v_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range (self.n_heads)])
        self.d_head = d_head
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim) (N,num_of_joints,2)
        # We got into shape (N, seq_len, n_heads, token_dim / n_heads)  (N,num_of_joints,1,2)
        # And come back to (N, seq_len, item_dim) (through concatenation)
        result = []
        for sequence in sequences :
            seq_result = []
            for head in range(self. n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence [:, head*self.d_head:(head+1)*self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
            return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class MyViTBlock(torch.nn.Module):
  def __init__(self, hidden_d, n_heads, mlp_ratio=4):
    super(MyViTBlock, self).__init__()
    self.hidden_d = hidden_d
    self.n_heads = n_heads

    self.norm1 = torch.nn.LayerNorm(hidden_d)
    # self.mhsa = MyMSA(hidden_d, n_heads)
    self.mhsa = Attention(hidden_d, n_heads, int(hidden_d/n_heads))
    self.norm2 = torch.nn.LayerNorm(hidden_d)
    self.mlp = torch.nn.Sequential(
        torch.nn.Linear(hidden_d, mlp_ratio*hidden_d),
        torch.nn.GELU(), 
        torch.nn.Linear(mlp_ratio * hidden_d, hidden_d )
    )

  def forward(self, x):
    out = x + self.mhsa(self.norm1(x))
    out = out + self.mlp(self.norm2(out))
    return out

class MyViT(torch.nn.Module):
  def __init__(self, chw=(1,17,2),  n_blocks=2, hidden_d=256, n_heads=4, out_d=3): #17 here might need to be changed to num_of_joints 
    #super constructer
    super(MyViT,self).__init__()

    #attributes
    self.chw = chw #(C,H,W)
    # self.n_patches = n_patches
    self.n_block = n_blocks
    self.n_heads = n_heads
    self.hidden_d = hidden_d


    # 1) Linear mapper 
    self.input_d = chw[2] #int(chw[0]*self.patch_size[0]*self.patch_size[1]) (4*4=16)
    self.linear_mapper = torch.nn.Linear(self.input_d, self.hidden_d)

    # 3) Positional embedding
    self.pos_embed = torch.nn.Parameter(torch.tensor(get_positional_embeddings(chw[1],self.hidden_d)))
    self.pos_embed.requires_grad = False

    # 4) Transformer encoder blocks
    self.blocks = torch.nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]) 

    # 5) Classification MLPK
    self.mlp = torch.nn.Sequential(
        torch.nn.Linear(self.hidden_d, int(self.hidden_d/2)), # 17*4=68
        torch.torch.nn.ReLU(), 
        torch.nn.Linear(int(self.hidden_d/2), out_d)#, #17*3=51
        # torch.nn.Tanh()#(dim=-1)
    )

  def forward(self,images):
    #Divising images into patches
    n, h, w = images.shape #n,c, h, w
    patches = images # patchify(images, self.n_patches)

    #Running linear layer tokenization 
    #Map the vector corresponding to each path to the hidden size dimension 
    tokens = self.linear_mapper(patches)

    #Adding positional embedding 
    pos_embed = self.pos_embed.repeat(n,1,1)
    out = tokens + pos_embed 

    #Transformer Blocks 
    for block in self.blocks: 
      out = block(out)
      
    out = self.mlp(out)

    return out  
