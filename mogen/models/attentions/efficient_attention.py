import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ATTENTIONS
from ..utils.stylization_block import StylizationBlock


@ATTENTIONS.register_module()
class EfficientSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_heads, dropout, time_embed_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim,
                                             dropout)

    def forward(self, x, src_mask, emb=None, **kwargs):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_heads
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        if self.time_embed_dim is None:
            y = x + y
        else:
            y = x + self.proj_out(y, emb)
        return y


@ATTENTIONS.register_module()
class EfficientCrossAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_heads, dropout,
                 time_embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb, cond_type=None, **kwargs):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_heads
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        if cond_type is None:
            key = F.softmax(key.view(B, N, H, -1), dim=1)
            # B, N, H, HD
            value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        else:
            text_cond_type = ((cond_type % 10) > 0).float().view(B, 1, 1)
            text_cond_type = text_cond_type.repeat(1, xf.shape[1], 1)
            key = key + (1 - text_cond_type) * -1000000
            key = F.softmax(key.view(B, N, H, -1), dim=1)
            value = self.value(self.text_norm(xf) * text_cond_type)
            value = value.view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


@ATTENTIONS.register_module()
class EfficientMixedAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_heads, dropout,
                 time_embed_dim):
        super().__init__()
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)

        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_text = nn.Linear(text_latent_dim, latent_dim)
        self.value_text = nn.Linear(text_latent_dim, latent_dim)
        self.key_motion = nn.Linear(latent_dim, latent_dim)
        self.value_motion = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb, src_mask, cond_type, **kwargs):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1] + x.shape[1]
        H = self.num_heads

        text_feat = xf
        # B, T, D
        query = self.query(self.norm(x)).view(B, T, H, -1)
        # B, N, D
        text_cond_type = (cond_type % 10 > 0).float()
        src_mask = src_mask.view(B, T, 1)
        key_text = self.key_text(self.text_norm(text_feat))
        key_text = key_text + (1 - text_cond_type) * -1000000
        key_motion = self.key_motion(self.norm(x)) + (1 - src_mask) * -1000000
        key = torch.cat((key_text, key_motion), dim=1)

        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = self.dropout(F.softmax(key.view(B, N, H, -1), dim=1))
        value = torch.cat((
            self.value_text(self.text_norm(text_feat)) * text_cond_type,
            self.value_motion(self.norm(x)) * src_mask,
        ),
                          dim=1).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y
