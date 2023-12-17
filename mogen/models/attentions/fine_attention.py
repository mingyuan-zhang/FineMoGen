import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ATTENTIONS
from ..utils.stylization_block import StylizationBlock

try:
    from tutel import moe as tutel_moe
    from tutel import net
except ImportError:
    pass


class MOE(nn.Module):

    def __init__(self, num_experts, topk, input_dim, ffn_dim, output_dim,
                 num_heads, max_seq_len, gate_type, gate_noise):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        try:
            data_group = net.create_groups_from_world(group_count=1).data_group
        except:
            data_group = None
        self.model = tutel_moe.moe_layer(gate_type={
            'type': gate_type,
            'k': topk,
            'fp32_gate': True,
            'gate_noise': gate_noise,
            'capacity_factor': 1.5
        },
                                         experts={
                                             'type': 'ffn',
                                             'count_per_node': num_experts,
                                             'hidden_size_per_expert': ffn_dim,
                                             'activation_fn':
                                             lambda x: F.gelu(x)
                                         },
                                         model_dim=input_dim,
                                         batch_prioritized_routing=True,
                                         is_gshard_loss=False,
                                         group=data_group)
        self.embedding = nn.Parameter(
            torch.randn(1, max_seq_len, num_heads, input_dim))

    def forward(self, x):
        B, T, H, D = x.shape
        x = x + self.embedding[:, :T, :, :]
        x = x.reshape(-1, D)
        y = self.proj(self.activation(self.model(x)))
        self.aux_loss = self.model.l_aux
        y = y.reshape(B, T, H, -1)
        return y


def get_ffn(latent_dim, ffn_dim):
    return nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.GELU(),
                         nn.Linear(ffn_dim, latent_dim))


@ATTENTIONS.register_module()
class SAMI(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_heads, num_text_heads,
                 num_experts, topk, gate_type, gate_noise, ffn_dim,
                 time_embed_dim, max_seq_len, max_text_seq_len, temporal_comb,
                 dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_text_heads = num_text_heads
        self.max_seq_len = max_seq_len

        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)

        self.sigma = nn.Parameter(torch.Tensor([100]))
        self.time = torch.arange(max_seq_len) / max_seq_len
        self.text_moe = MOE(num_experts, topk, text_latent_dim,
                            text_latent_dim * 4, 2 * latent_dim,
                            num_text_heads, max_text_seq_len, gate_type,
                            gate_noise)
        self.motion_moe = MOE(num_experts, topk, latent_dim, latent_dim * 4,
                              3 * latent_dim, num_heads, max_seq_len,
                              gate_type, gate_noise)
        self.key_motion = nn.Parameter(torch.randn(max_seq_len, latent_dim))
        self.body_weight = nn.Parameter(torch.randn(num_heads, num_heads))

        self.template_s = get_ffn(latent_dim, ffn_dim)
        self.template_v = get_ffn(latent_dim, ffn_dim)
        self.template_a = get_ffn(latent_dim, ffn_dim)
        self.template_j = get_ffn(latent_dim, ffn_dim)
        self.template_t = nn.Sequential(nn.Linear(latent_dim, ffn_dim),
                                        nn.GELU(), nn.Linear(ffn_dim, 1))
        self.t_sigma = nn.Parameter(torch.Tensor([1]))
        self.proj_out = StylizationBlock(latent_dim * num_heads,
                                         time_embed_dim, dropout)
        self.temporal_comb = temporal_comb

    def forward(self, x, xf, emb, src_mask, cond_type, motion_length,
                num_intervals, **kwargs):
        """
        x: B, T, D
        xf: B, N, P
        """
        B, T, D = x.shape
        N = xf.shape[1] + x.shape[1]
        H = self.num_heads
        L = self.latent_dim

        x = x.reshape(B, T, H, -1)
        text_feat = xf.reshape(B, xf.shape[1], self.num_text_heads, -1)
        text_feat = self.text_moe(self.text_norm(text_feat))
        motion_feat = self.motion_moe(self.norm(x))

        body_weight = F.softmax(self.body_weight, dim=1)
        body_value = motion_feat[:, :, :, :L]
        body_feat = torch.einsum('hl,bnld->bnhd', body_weight, body_value)
        body_feat = body_feat.reshape(B, T, D)

        # B, N, D
        text_cond_type = (cond_type % 10 > 0).float().unsqueeze(-1)
        src_mask = src_mask.view(B, T, 1, 1)

        key_text = text_feat[:, :, :, :L].contiguous()
        key_text = key_text + (1 - text_cond_type) * -1000000
        if self.num_text_heads == 1:
            key_text = key_text.repeat(1, 1, H, 1)
        key_motion = motion_feat[:, :, :, L:2 * L].contiguous()
        key_motion = key_motion + (1 - src_mask) * -1000000
        key = torch.cat((key_text, key_motion), dim=1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)

        value_text = text_feat[:, :, :, L:].contiguous() * text_cond_type
        if self.num_text_heads == 1:
            value_text = value_text.repeat(1, 1, H, 1)
        value_motion = motion_feat[:, :, :, 2 * L:].contiguous() * src_mask
        value = torch.cat((value_text, value_motion), dim=1).view(B, N, H, -1)

        # B, H, d, l
        template = torch.einsum('bnhd,bnhl->bhdl', key, value)
        template_t_feat = self.template_t(template)
        template_t = F.sigmoid(template_t_feat / self.t_sigma)
        template_t = template_t * motion_length.view(B, 1, 1, 1)
        template_t = template_t / self.max_seq_len
        org_t = self.time[:T].type_as(x)

        NI = num_intervals
        t = org_t.clone().view(1, 1, -1, 1, 1).repeat(B // NI, NI, 1, 1, 1)
        template_t = template_t.view(-1, NI, H, L)
        motion_length = motion_length.view(-1, NI)
        for b_ix in range(B // NI):
            sum_frames = 0
            for i in range(NI):
                t[b_ix, i] += sum_frames / self.max_seq_len
                template_t[b_ix, i] += sum_frames / self.max_seq_len
                sum_frames += motion_length[b_ix, i]
        template_t = template_t.permute(0, 2, 1, 3)
        template_t = template_t.unsqueeze(1).repeat(1, NI, 1, 1, 1)
        template_t = template_t.reshape(B, 1, H, -1)
        time_delta = t.view(B, -1, 1, 1) - template_t
        time_delta = time_delta * self.max_seq_len
        time_sqr = time_delta * time_delta
        time_coef = F.softmax(-time_sqr / self.sigma, dim=-1)

        template = template.view(-1, NI, H, L, L)
        template = template.permute(0, 2, 1, 3, 4).unsqueeze(1)
        template = template.repeat(1, NI, 1, 1, 1, 1)
        template = template.reshape(B, H, -1, L)
        # Taylor expansion
        template_s = self.template_s(template)  # state
        template_v = self.template_v(template)  # velocity
        template_a = self.template_a(template)  # acceleration
        template_j = self.template_j(template)  # jerk
        template_t = template_t.view(B, H, -1, 1)
        template_a0 = template_s - template_v * template_t + \
            template_a * template_t * template_t - \
            template_j * template_t * template_t * template_t
        template_a1 = template_v - 2 * template_a * template_t + \
            3 * template_j * template_t * template_t
        template_a2 = template_a - 3 * template_j * template_t
        template_a3 = template_j
        a0 = torch.einsum('bnhd,bhdl->bnhl', time_coef,
                          template_a0).reshape(B, T, D)
        a1 = torch.einsum('bnhd,bhdl->bnhl', time_coef,
                          template_a1).reshape(B, T, D)
        a2 = torch.einsum('bnhd,bhdl->bnhl', time_coef,
                          template_a2).reshape(B, T, D)
        a3 = torch.einsum('bnhd,bhdl->bnhl', time_coef,
                          template_a3).reshape(B, T, D)
        t = t.view(B, -1, 1)
        y_t = a0 + a1 * t + a2 * t * t + a3 * t * t * t
        y_s = body_feat
        y = x.reshape(B, T, D) + self.proj_out(y_s + y_t, emb)
        if self.training:
            self.aux_loss = self.text_moe.aux_loss + self.motion_moe.aux_loss
            mu = template_t_feat.squeeze(-1).mean(dim=-1)
            logvar = torch.log(template_t_feat.squeeze(-1).std(dim=-1))
            self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
                                            logvar.exp())
        return y
