import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ManualAttention(nn.Module):
    """
    Standard ViT Attention patched to be HVP-safe.
    Replaces fused kernels with pure torch.matmul to preserve the 2nd-order graph.
    """
    def __init__(self, dim, num_heads=3, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x,attn_mask=None):
        B, N, C = x.shape
        # Split QKV: [3, Batch, Heads, Tokens, Head_Dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Pure MatMul Attention (HVP-Safe)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

def get_proper_vit_tiny(num_classes=100):
    """
    Loads pre-trained ViT-Tiny (192 dim, 3 heads, 12 blocks).
    Adapted for 32x32 images with Patch Size 4.
    """
    # Load standard ViT-Tiny backbone
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    
    # 1. Patch Attention layers to be HVP-Safe
    for block in model.blocks:
        dim = block.attn.qkv.in_features
        num_heads = block.attn.num_heads
        block.attn = ManualAttention(dim, num_heads=num_heads, qkv_bias=True)
        
    # 2. Adjust Patch Embedding for CIFAR (32x32 -> 8x8 grid)
    model.patch_embed.proj = nn.Conv2d(3, 192, kernel_size=4, stride=4)
    model.patch_embed.img_size = (32, 32)
    model.patch_embed.patch_size = (4, 4)
    model.patch_embed.grid_size = (8, 8)
    model.patch_embed.num_patches = 64
    
    # 3. Interpolate Positional Embeddings (197 -> 65)
    num_extra_tokens = 1 
    pos_tokens = model.pos_embed[:, num_extra_tokens:] # [1, 196, 192]
    pos_tokens = pos_tokens.reshape(1, 14, 14, 192).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(8, 8), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, 64, 192)
    
    model.pos_embed = nn.Parameter(torch.cat((model.pos_embed[:, :num_extra_tokens], pos_tokens), dim=1))
    
    # 4. Final Classification Head
    model.head = nn.Linear(192, num_classes)
    
    return model