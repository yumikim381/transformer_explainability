""" 
So key idea is that we only need to watch out for skip connection(add layer) and matrix multiplication for relevance propagation 
"""
import torch
import torch.nn as nn
from einops import rearrange
from modules.layers_ours import *

from baselines.ViT.helpers import load_pretrained
from baselines.ViT.weight_init import trunc_normal_
from baselines.ViT.layer_helpers import to_2tuple


 # For each layer they defined a function in "layer_ours.py"

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        #timm/vit_base_patch16_224.orig_in21k_ft_in1k
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}



#classification head 
# This is the two layer MLP head to classify the image based on [CLS] token embedding.
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    # Partial derivation of the operation we are doing in this layer 
    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam

#Multi-head attention mechanism
class Attention(nn.Module):
    """
    qkv_bias: Boolean indicating whether to include bias terms in the query, key, and value projections.
    attn_drop, proj_drop: Dropout rates for attention weights and output projection.
    num_heads, scale: The number of heads and scaling factor used to normalize the dot products during attention computation.
    qkv: A linear layer that expands input dimension dim to three times its size for generating query, key, and value matrices.
    attn_drop, proj_drop: Dropout layers for attention weights and the final projected output.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients
    """
        Shape Handling: Computes number of batches b, number of tokens n, and reshapes x for multi-head attention.
        Query, Key, Value Computation: Uses the qkv linear transformation and reshapes the output to separate queries, keys, and values.
        Dot Product and Scaling: Computes dot products between queries and keys, scales the results, and applies softmax to get the attention weights.
        """
    def forward(self, x):
        # third parameter is feature size 
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        """
        The projection output is reshaped and split into three separate tensors for queries (q), keys (k), and values (v). 
        The reshaping arranges each tensor to be in the format suitable for multi-head attention, 
        where h is the number of heads and d is the dimension per head.
        """
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)
        """
        Computes the dot product of queries and keys, which forms the basis of the attention mechanism. 
        The result is scaled down by the square root of the dimension per head to stabilize gradients during training (preventing values from becoming too large).
        """
        dots = self.matmul1([q, k]) * self.scale
        """
        Softmax is applied to the scaled dot products to obtain attention weights, 
        which represent the importance of each key to each query.
        """
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        """
        reshaped to separate the concatenated head dimensions back into separate heads (h), 
        preparing it for propagation through the multi-head structure.
        """
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        """
        This splits the relevance into components corresponding to the attention weights (cam1) 
        and the values (cam_v) after being combined in the attention calculation (self.matmul2 corresponds to attn = A*V).
        """
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        """
        https://github.com/hila-chefer/Transformer-Explainability/issues/10
        we divide each of the cams by 2 after the matmul operation, which is identical to applying our normalization.
        It is important to notice that this is the case for matmul only, for add layers, 
        the sums of the two relevancies may not be equal, as we point out in the paper.
        => For add layers its in layers implementation 

        """
        cam1 /= 2
        cam_v /= 2
        # Save Intermediate Relevances:
        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)

## transformer consists of blocks 
"""
each block b is composed of self-attention, skip con- nections, 
and additional linear and normalization layers in a certain assembly
"""
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Residual connection 
        self.add1 = Add()
        self.add2 = Add()
        """
        duplicate the input tensor so that one copy can be transformed while the other is passed through unchanged, 
        supporting the residual connections.
        """
        
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        #Pass x2 through norm and attention layer and add the result to x1 
        x = self.add1([x1, self.attn(self.norm1(x2))])
        # clone the subresult again and do the same 
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        #Just reversing the actions from VIT
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PatchEmbed(nn.Module):
    """
    - responsible for converting an input image into a sequence of flattened, embedded patches
    Get patch embeddings 
    -splits the image into patches of equal size and do a linear transformation on the flattened pixels for each patch.
    - implement through a convolution layer, because it's simpler to implement.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        #in_chanss is the number of channels in the input image (3 for rgb)
        # transformer embeddings size for embed_dim
        super().__init__()
        """
        Both img_size and patch_size are converted to tuples using to_2tuple to ensure they are expressed as (height, width)
        """
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        """
        num_patches is calculated based on the division of the image dimensions by the patch dimensions. 
        This computes how many patches the image will be divided into across its width and height."""
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        """
        We create a convolution layer with a kernel size and and stride length equal to patch size. 
        This is equivalent to splitting the image into patches and doing a linear transformation on each patch.
        """ 
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        """
        output of this layer will be a tensor where each "pixel" in the output corresponds to one patch of the input.
        Each patch is independently transformed to the embedding dimension embed_dim.
        """

    def forward(self, x):
        #x is the input image of shape [batch_size, channels, height, width]
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #Apply convolution layer
        """
        The output of the convolution layer is first flattened starting from the third dimension. 
        This flattening operation converts the 2D patch embeddings into a 1D format per patch.
        It then transposes the second and third dimensions, which adjusts the tensor shape 
        to [batch_size, num_patches, embed_dim]. This results in a sequence of embedded patches suitable 
        for processing by the subsequent transformer blocks.
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        # [batch_size, num_patches, embed_dim]. The transpose changes this to [batch_size, embed_dim, num_patches].
        cam = cam.transpose(1,2)
        """
        This step essentially maps the flattened patch embeddings back into their original positions in the 2D grid of the image.
        """
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    """
    all_layer_matrices: A list of tensors where each tensor is an attention matrix from a consecutive transformer layer.
    start_layer: The starting layer from which to begin the calculation of rollout attention. The default value is 0, which indicates starting from the first layer.
    num_tokens: The number of tokens or sequence elements in the attention matrix, determined by the second dimension of the first layer's attention matrix.
    batch_size: The batch size, determined by the first dimension of the first layer's attention matrix.
    eye - identity matrix , This identity matrix is expanded to match the batch size and moved to the same device as the attention matrices to ensure compatibility.
    """
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    """The identity matrix is added to each attention matrix in all_layer_matrices. 
    This step incorporates the residual connection typically used in transformer architectures,
    ensuring that each token retains a portion of its original state besides what is transformed by attention.
    """
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    """
    The function initializes the joint_attention with the attention matrix from the start_layer.
    It then iteratively computes the product of this joint_attention matrix 
    with each subsequent attention matrix using batch matrix multiplication (bmm)
    """
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0.):
        #Initializes the VisionTransformer class inheriting from nn.Module.
        # number of layers (depth) - number of transformer blocks 
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        """
        pos_embed
        1: This represents a single batch of positional embeddings that can be expanded to match any batch size during training.
        num_patches + 1: The total number of patches that the image is split into, plus one additional slot for the [CLS] token, which is used for classification. Each patch and the [CLS] token will have a unique positional embedding.
        embed_dim: The dimensionality of each positional embedding vector, which matches the dimensionality of the patch embeddings and the [CLS] token embedding.
        """
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        """
        1: One batch of [CLS] token embeddings, which can be expanded to match the batch size during training.
        1: There is only one [CLS] token per image or per sequence.
        embed_dim: The dimensionality of the [CLS] token embedding, which is the same as the dimensionality of the positional embeddings and the patch embeddings.
        """
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        """
         series of transformer blocks (self.blocks) is created, 
         where each block consists of multi-head self-attention and MLP layers
        """
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        self.norm = LayerNorm(embed_dim)
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head = Linear(embed_dim, num_classes)

        # FIXME not quite sure what the proper weight init is supposed to be,
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        """
        trunc_normal_ is a function used to initialize the values of a tensor with values drawn from a truncated normal distribution. 
        This type of distribution is a normal distribution where values whose magnitude is more than a certain number of standard deviations from the mean are dropped and redrawn
        """
        trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## IndexSelect for selecting cls tokens 
        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        """
        Prepends the [CLS] token and adds positional embeddings to the sequence.
        """
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])
        """
        Registers a hook to save input gradients for interpretability.
        """
        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        """
        This indicates that the method should select the element at index 0 along dimension 1 for each item in the batch. 
        The [CLS] token is usually positioned at the start of the sequence, hence the index 0.
        """
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        """
        [batch_size, 1, feature_dim] to [batch_size, feature_dim],
        """
        x = x.squeeze(1)
        """
        Passing through the Classification Head
        """
        x = self.head(x)
        return x

    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                #Cam is relevance score and grad is gradients of attention 
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                #Reshape to ensure they align properly for element-wise multiplication. 
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                #Equation 13 in paper 
                cam = grad * cam
                #clamp method in PyTorch is used to limit the values in a tensor to a specified range. 
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            #So at the end, cams should be "A list of tensors where each tensor is an attention matrix from a consecutive transformer layer"
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))

    return model

def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

'''def swin_tiny_patch4_window7_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=4, embed_dim=96, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    (
        image_size=224, patch_size=4, num_channels=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, drop_path_rate=0.1, hidden_act="gelu", use_absolute_embeddings=False, initializer_range=0.02,
        layer_norm_eps=1e-5, encoder_stride=32, out_features=None, out_indices=None,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/swin_tiny_patch4_window7_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model'''