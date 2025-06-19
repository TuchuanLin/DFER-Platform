import math
import torch
import torch.nn as nn

class TemporalHiLoWithPositionalEncoding(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5, seq_len=128):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        # Self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim

        # Self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim

        self.scale = qk_scale or head_dim ** -0.5

        # Position Encoding
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

        # Lo-Fi Attention
        if self.l_heads > 0:
            self.l_q = nn.Linear(dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # Hi-Fi Attention
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

        # Temporal Attention
        self.temporal_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def hifi(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.window_size, W // self.window_size
        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.window_size, w_group, self.window_size, C).transpose(2, 3)
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.window_size, self.window_size, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.window_size, w_group * self.window_size, self.h_dim)
        return self.h_proj(x)

    def lofi(self, x):
        B, H, W, C = x.shape
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        return self.l_proj(x)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x + self.position_embedding[:, :N, :]  # Add position encoding
        x = x.reshape(B, H, W, C)

        hifi_out = self.hifi(x) if self.h_heads > 0 else torch.zeros_like(x)
        lofi_out = self.lofi(x) if self.l_heads > 0 else torch.zeros_like(x)

        # Concatenate Hi-Fi and Lo-Fi attention results
        x = torch.cat((hifi_out, lofi_out), dim=-1).reshape(B, N, C)

        # Apply temporal attention
        x, _ = self.temporal_attn(x, x, x)
        return x
    
####################################################################################################

class HiLo(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.dim = dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim

        # local window size. The `s` in our paper.
        self.ws = window_size

        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2] # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = x.reshape(B, H, W, C)

        if self.h_heads == 0:
            x = self.lofi(x)
            return x.reshape(B, N, C)

        if self.l_heads == 0:
            x = self.hifi(x)
            return x.reshape(B, N, C)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, N, C)

        return x

    def flops(self, H, W):
        # pad the feature map when the height and width cannot be divided by window size
        Hp = self.ws * math.ceil(H / self.ws)
        Wp = self.ws * math.ceil(W / self.ws)

        Np = Hp * Wp

        # For Hi-Fi
        # qkv
        hifi_flops = Np * self.dim * self.h_dim * 3
        nW = (Hp // self.ws) * (Wp // self.ws)
        window_len = self.ws * self.ws
        # q @ k and attn @ v
        window_flops = window_len * window_len * self.h_dim * 2
        hifi_flops += nW * window_flops
        # projection
        hifi_flops += Np * self.h_dim * self.h_dim

        # for Lo-Fi
        # q
        lofi_flops = Np * self.dim * self.l_dim
        kv_len = (Hp // self.ws) * (Wp // self.ws)
        # k, v
        lofi_flops += kv_len * self.dim * self.l_dim * 2
        # q @ k and attn @ v
        lofi_flops += Np * self.l_dim * kv_len * 2
        # projection
        lofi_flops += Np * self.l_dim * self.l_dim

        return hifi_flops + lofi_flops
    
####################################################################################################    
import math
import torch
import torch.nn as nn

class BiDirectionalCausalHiLo(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 window_size=2, alpha=0.5, seq_len=128):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        # Self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim

        # Self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim

        self.scale = qk_scale or head_dim ** -0.5

        # Position Encoding
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

        # Lo-Fi Attention
        if self.l_heads > 0:
            self.l_q = nn.Linear(dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # Hi-Fi Attention
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

        # Forward Causal Attention
        self.forward_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.forward_causal_mask = self.generate_causal_mask(seq_len, forward=True)

        # Backward Causal Attention
        self.backward_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.backward_causal_mask = self.generate_causal_mask(seq_len, forward=False)

    def generate_causal_mask(self, seq_len, forward=True):
        """Generate a causal mask for attention to ensure each time step attends only to valid tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1 if forward else 0).transpose(0, 1)
        if not forward:
            mask = torch.tril(mask)
        return mask.masked_fill(mask == 1, float('-inf'))

    def hifi(self, x):
        B, T, C = x.shape
        H = int(math.sqrt(T))
        x = x.reshape(B, H, H, C)  # Reshape into spatial dimensions for Hi-Lo

        h_group, w_group = H // self.window_size, H // self.window_size
        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.window_size, w_group, self.window_size, C).transpose(2, 3)
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.window_size, self.window_size, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, H, H, self.h_dim)
        return self.h_proj(x.reshape(B, T, self.h_dim))

    def forward(self, x):
        B, T, C = x.shape

        # Add temporal positional encoding
        x = x + self.position_embedding[:, :T, :]

        # Step 1: Forward causal attention
        forward_out, _ = self.forward_attn(x, x, x, attn_mask=self.forward_causal_mask.to(x.device))

        # Step 2: Backward causal attention
        backward_out, _ = self.backward_attn(x, x, x, attn_mask=self.backward_causal_mask.to(x.device))

        # Step 3: Concatenate both outputs
        bi_out = torch.cat([forward_out, backward_out], dim=-1)  # [B, T, 2D]

        return bi_out

if __name__ == '__main__':
   # block = TemporalHiLoWithPositionalEncoding(dim=128, seq_len=128)
    block = BiDirectionalCausalHiLo(dim=128, seq_len=128)
    input = torch.rand(32, 128, 128)  # input shape (B, N, C)
    output = block(input)  # H = 16, W = 8, H * W = N
    print(input.size())
    print(output.size())