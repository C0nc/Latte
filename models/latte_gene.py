import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb

class CellTypeEmbedder(nn.Module):
    """Embeds cell type labels into vector representations."""
    def __init__(self, num_cell_types, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.embedding_table = nn.Embedding(num_cell_types + 1, hidden_size)  # +1 for classifier-free guidance
        self.num_cell_types = num_cell_types
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_cell_types, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c shape: [batch_size, num_timepoints, hidden_size]
        # Get modulation parameters from the first timepoint
        c_hidden = c[:, 0, :]  # Use first timepoint's condition
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_hidden).chunk(6, dim=1)
        
        # Apply attention block
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # Apply MLP block
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """Final layer of GeneLatte."""
    def __init__(self, hidden_size, out_features):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_features, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class GeneLatte(nn.Module):
    """
    Diffusion model for single-cell gene expression time series data.
    """
    def __init__(
        self,
        num_genes=10000,           # Number of genes in expression data
        num_timepoints=16,         # Number of timepoints
        num_cell_types=100,        # Number of cell types
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cell_type_dropout_prob=0.1,
        learn_sigma=True
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_genes = num_genes
        self.out_features = num_genes * 2 if learn_sigma else num_genes
        self.num_timepoints = num_timepoints
        self.hidden_size = hidden_size
        
        # Embedding layers
        self.gene_embedder = nn.Linear(num_genes, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cell_type_embedder = CellTypeEmbedder(num_cell_types, hidden_size, cell_type_dropout_prob)
        
        # Positional embeddings for timepoints
        self.time_embed = nn.Parameter(torch.zeros(1, num_timepoints, hidden_size), requires_grad=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) 
            for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, self.out_features)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize time_embed with sinusoidal embedding
        time_embed = self._get_1d_sincos_pos_embed(self.hidden_size, self.num_timepoints)
        self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize cell type embedding
        nn.init.normal_(self.cell_type_embedder.embedding_table.weight, std=0.02)

        # Zero-out modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _get_1d_sincos_pos_embed(self, embed_dim, length):
        """Generate sinusoidal positional embeddings."""
        pos = np.arange(length)[:, np.newaxis]
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        pos_emb = pos * omega
        pos_emb = np.concatenate([np.sin(pos_emb), np.cos(pos_emb)], axis=1)
        return pos_emb

    def forward(self, x, t, cell_types, use_fp16=False):
        """
        Forward pass of GeneLatte.
        x: (batch_size, num_timepoints, num_genes) tensor of gene expression data
        t: (batch_size,) tensor of diffusion timesteps
        cell_types: (batch_size,) tensor of cell type labels
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)
        
        # Embed gene expression data
        x = self.gene_embedder(x)  # Shape: (batch_size, num_timepoints, hidden_size)
        
        # Add temporal positional embeddings
        x = x + self.time_embed
        
        # Get timestep and cell type embeddings
        t_emb = self.t_embedder(t, use_fp16=use_fp16)
        cell_type_emb = self.cell_type_embedder(cell_types, self.training)
        
        # Combine condition embeddings
        c = t_emb + cell_type_emb
        
        # Expand condition embedding for each timepoint
        c = repeat(c, 'b d -> b n d', n=self.num_timepoints)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c[:, 0, :])  # Use first timepoint's embedding
        
        # Reshape output
        x = x.view(-1, self.num_timepoints, self.out_features)
        
        return x

    def forward_with_cfg(self, x, t, cell_types, cfg_scale, use_fp16=False):
        """
        Forward pass with classifier-free guidance.
        """
        # Run both conditional and unconditional forward passes
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        cell_types_null = torch.ones_like(cell_types[:len(cell_types) // 2]) * self.cell_type_embedder.num_cell_types
        combined_cell_types = torch.cat([cell_types[:len(cell_types) // 2], cell_types_null])
        
        model_out = self.forward(combined, t, combined_cell_types, use_fp16=use_fp16)
        
        # Split predictions and compute guided output
        eps, rest = model_out[:, :, :self.num_genes], model_out[:, :, self.num_genes:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        
        return torch.cat([eps, rest], dim=2)

def GeneLatte_XL(**kwargs):
    return GeneLatte(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def GeneLatte_L(**kwargs):
    return GeneLatte(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def GeneLatte_B(**kwargs):
    return GeneLatte(depth=12, hidden_size=768, num_heads=12, **kwargs)

def GeneLatte_S(**kwargs):
    return GeneLatte(depth=12, hidden_size=384, num_heads=6, **kwargs)



if __name__ == '__main__':
    # Example usage
    batch_size = 32
    num_genes = 10000
    num_timepoints = 16
    num_cell_types = 100
    
    model = GeneLatte_XL(
        num_genes=num_genes,
        num_timepoints=num_timepoints,
        num_cell_types=num_cell_types
    )
    
    # Create sample inputs
    gene_expression = torch.randn(batch_size, num_timepoints, num_genes)
    timesteps = torch.randint(0, 1000, (batch_size,))
    cell_types = torch.randint(0, num_cell_types, (batch_size,))
    
    # Regular forward pass
    output = model(gene_expression, timesteps, cell_types)
    
    print(output.shape)