import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from modules.layers_ours import *
#https://python.plainenglish.io/swin-transformer-from-scratch-in-pytorch-31275152bf03

"""
“It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT. Each patch is treated as a “token” and its feature is set as a concatenation of the raw pixel RGB values. In our implementation, we use a patch size of 4 × 4 and thus the feature dimension of each patch is 4 × 4 × 3 = 48. 
A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension (denoted as C).”

Why we do convolution:ViT style patch partition + linear embedding step is to use a convolution where kernel size = stride = patch size and output channels = C
"""

class SwinEmbedding(nn.Module):

    '''
    input shape -> (b,c,h,w)
    output shape -> (b, (h/4 * w/4), C)
    '''

    def __init__(self, patch_size=4, C=96):
        super().__init__()
        self.linear_embedding = Conv2d(3, C, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = LayerNorm(C)
        self.relu = ReLU()

    def forward(self,x):
        x = self.linear_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.relu(self.layer_norm(x))
        return x
    def relprops(self, cam, **kwargs):
        cam = self.relu.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b (h w) c-> b c h w ')
        cam = self.linear_embedding.relprop(cam, **kwargs)
        return cam
        

"""
To produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper. 
The first patch merging layer concatenates the features of each group of 2 × 2 neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of 2×2 = 4 (2× downsampling of resolution), and the output dimension is set to 2C.

What it does:
The patch merging layer is straightforward. 
We initialize a linear layer with 4C input channels to 2C output channels and initialize a layer norm with the output embedding size. 
In our forward function we use einops rearrange to reshape our tokens from 2x2xC to 1x1x4C. 
We finish by passing our inputs through the linear projection and layer norm.
"""

class PatchMerging(nn.Module):

    '''
    input shape -> (b, (h*w), C)
    output shape -> (b, (h/2 * w/2), C*2)
    '''

    def __init__(self, C):
        super().__init__()
        self.linear = Linear(4*C, 2*C)
        self.layer_norm = LayerNorm(2*C)

    def forward(self, x):
        height = width = int(math.sqrt(x.shape[1])/2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)', s1=2, s2=2, h=height, w=width)
        return self.layer_norm(self.linear(x))
    
    def relprops(self, cam, **kwargs):
        cam=self.linear.relprop(cam, **kwargs)
        cam = self.layer_norm.relprop(cam, **kwargs)
        height = width = int(math.sqrt(cam.shape[1])/2)
        cam = rearrange(cam, 'b (h w) (s2 s1 c) -> b (h s1 w s2) c ', s1=2, s2=2, h=height, w=width)

class ShiftedWindowMSA(nn.Module):

    '''
    input shape -> (b,(h*w), C)
    output shape -> (b, (h*w), C)
    '''

    def __init__(self, embed_dim, num_heads, window_size=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.proj1 = nn.Linear(embed_dim, 3*embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        h_dim = self.embed_dim / self.num_heads
        height = width = int(math.sqrt(x.shape[1]))
        x = self.proj1(x)

        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)
        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H=self.num_heads, m1=self.window_size, m2=self.window_size)
        
        '''
            H = # of Attention Heads
            h,w = # of windows vertically and horizontally
            (m1 m2) = total size of each window
            E = head dimension
            K = 3 = a constant to break our matrix into 3 Q,K,V matricies 
        '''

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        att_scores = (Q @ K.transpose(4,5)) / math.sqrt(h_dim)
        att = F.softmax(att_scores, dim=-1) @ V

        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=self.window_size, m2=self.window_size)
        x = rearrange(x, 'b h w c -> b (h w) c')

        return self.proj2(x)
