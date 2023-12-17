import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ATTENTIONS
from ..utils.stylization_block import StylizationBlock


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


@ATTENTIONS.register_module()
class SemanticsModulatedAttention(nn.Module):

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

        self.retr_norm1 = nn.LayerNorm(2 * latent_dim)
        self.retr_norm2 = nn.LayerNorm(latent_dim)
        self.key_retr = nn.Linear(2 * latent_dim, latent_dim)
        self.value_retr = zero_module(nn.Linear(latent_dim, latent_dim))

        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb, src_mask, cond_type, re_dict=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        re_motion = re_dict['re_motion']
        re_text = re_dict['re_text']
        re_mask = re_dict['re_mask']
        re_mask = re_mask.reshape(B, -1, 1)
        N = xf.shape[1] + x.shape[1] + re_motion.shape[1] * re_motion.shape[2]
        H = self.num_heads
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        text_cond_type = (cond_type % 10 > 0).float()
        retr_cond_type = (cond_type // 10 > 0).float()
        re_text = re_text.repeat(1, 1, re_motion.shape[2], 1)
        re_feat_key = torch.cat((re_motion, re_text), dim=-1)
        re_feat_key = re_feat_key.reshape(B, -1, 2 * D)

        key_text = self.key_text(self.text_norm(xf))
        key_text += (1 - text_cond_type) * -1000000
        key_retr = self.key_retr(self.retr_norm1(re_feat_key))
        key_retr += (1 - retr_cond_type) * -1000000 + (1 - re_mask) * -1000000
        key_motion = self.key_motion(self.norm(x))
        key_motion += (1 - src_mask) * -1000000
        key = torch.cat((key_text, key_retr, key_motion), dim=1)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        re_feat_value = re_motion.reshape(B, -1, D)
        value_text = self.value_text(self.text_norm(xf)) * text_cond_type
        value_retr = self.value_retr(self.retr_norm2(re_feat_value))
        value_retr = value_retr * retr_cond_type * re_mask
        value_motion = self.value_motion(self.norm(x)) * src_mask
        value = torch.cat((value_text, value_retr, value_motion),
                          dim=1).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


@ATTENTIONS.register_module()
class DualSemanticsModulatedAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_heads, dropout,
                 time_embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)

        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_text = nn.Linear(text_latent_dim, latent_dim)
        self.value_text = nn.Linear(text_latent_dim, latent_dim)
        self.key_motion = nn.Linear(latent_dim, latent_dim)
        self.value_motion = nn.Linear(latent_dim, latent_dim)
        self.key_inter = nn.Linear(latent_dim, latent_dim)
        self.value_inter = nn.Linear(latent_dim, latent_dim)

        self.retr_norm1 = nn.LayerNorm(2 * latent_dim)
        self.retr_norm2 = nn.LayerNorm(latent_dim)
        self.key_retr = nn.Linear(2 * latent_dim, latent_dim)
        self.value_retr = zero_module(nn.Linear(latent_dim, latent_dim))

        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb, src_mask, cond_type, re_dict=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        x1 = x[:, :, :self.latent_dim].contiguous()
        x2 = x[:, :, self.latent_dim:].contiguous()
        B, T, D = x1.shape
        re_motion = re_dict['re_motion']
        re_text = re_dict['re_text']
        re_mask = re_dict['re_mask']
        re_mask = re_mask.reshape(B, -1, 1)
        N = xf.shape[1] + x.shape[1] * 2 + \
            re_motion.shape[1] * re_motion.shape[2]
        H = self.num_heads
        # B, T, D
        query1 = self.query(self.norm(x1))
        query2 = self.query(self.norm(x2))
        # B, N, D
        text_cond_type = (cond_type % 10 > 0).float()
        retr_cond_type = (cond_type // 10 > 0).float()
        re_text = re_text.repeat(1, 1, re_motion.shape[2], 1)
        re_feat_key = torch.cat((re_motion, re_text), dim=-1)
        re_feat_key = re_feat_key.reshape(B, -1, 2 * D)

        key_text = self.key_text(self.text_norm(xf))
        key_text += (1 - text_cond_type) * -1000000
        key_retr = self.key_retr(self.retr_norm1(re_feat_key))
        key_retr += (1 - retr_cond_type) * -1000000 + (1 - re_mask) * -1000000
        key_motion1 = self.key_motion(self.norm(x1))
        key_motion1 += (1 - src_mask) * -1000000
        key_motion2 = self.key_motion(self.norm(x2))
        key_motion2 += (1 - src_mask) * -1000000
        key_inter1 = self.key_inter(self.norm(x2))
        key_inter1 += (1 - src_mask) * -1000000
        key_inter2 = self.key_inter(self.norm(x1))
        key_inter2 += (1 - src_mask) * -1000000
        key1 = torch.cat((key_text, key_retr, key_motion1, key_inter1), dim=1)
        key2 = torch.cat((key_text, key_retr, key_motion2, key_inter2), dim=1)
        query1 = F.softmax(query1.view(B, T, H, -1), dim=-1)
        query2 = F.softmax(query2.view(B, T, H, -1), dim=-1)
        key1 = F.softmax(key1.view(B, N, H, -1), dim=1)
        key2 = F.softmax(key2.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        re_feat_value = re_motion.reshape(B, -1, D)
        value_text = self.value_text(self.text_norm(xf)) * text_cond_type
        value_retr = self.value_retr(self.retr_norm2(re_feat_value))
        value_retr = value_retr * retr_cond_type * re_mask
        value_motion1 = self.value_motion(self.norm(x1)) * src_mask
        value_motion2 = self.value_motion(self.norm(x2)) * src_mask
        value_inter1 = self.value_inter(self.norm(x2)) * src_mask
        value_inter2 = self.value_inter(self.norm(x1)) * src_mask
        value1 = torch.cat((
            value_text,
            value_retr,
            value_motion1,
            value_inter1,
        ),
                           dim=1).view(B, N, H, -1)
        value2 = torch.cat((
            value_text,
            value_retr,
            value_motion2,
            value_inter2,
        ),
                           dim=1).view(B, N, H, -1)
        # B, H, HD, HD
        attention1 = torch.einsum('bnhd,bnhl->bhdl', key1, value1)
        attention2 = torch.einsum('bnhd,bnhl->bhdl', key2, value2)
        y1 = torch.einsum('bnhd,bhdl->bnhl', query1, attention1)
        y2 = torch.einsum('bnhd,bhdl->bnhl', query2, attention2)
        y1 = x1 + self.proj_out(y1.reshape(B, T, D), emb)
        y2 = x2 + self.proj_out(y2.reshape(B, T, D), emb)
        y = torch.cat((y1, y2), dim=-1)
        return y
