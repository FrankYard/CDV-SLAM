
import torch
import torch.nn as nn
from typing import Callable, List, Optional

from .attention import Attention

class GatedAttention(nn.Module):
    def __init__(self, dim, nhead, attention):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid())

        self.encoder = EncoderLayer(dim, nhead=nhead, attention=attention)

    def forward(self, x, encoding):
        return x + self.gate(x) * self.encoder(x, encoding=encoding, only_residual=True)

################### From LightGlue ######################

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


############# From Mickey #############

class EncoderLayer(nn.Module):
    """
        Transformer encoder layer containing the linear self and cross-attention, and the epipolar attention.
        Arguments:
            d_model: Feature dimension of the input feature maps (default: 128d).
            nhead: Number of heads in the multi-head attention.
            attention: Type of attention for the common transformer block. Options: linear, full.
    """
    def __init__(self, d_model, nhead, attention='linear'):
        super(EncoderLayer, self).__init__()

        # Transformer encoder layer parameters
        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention definition
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(attention=attention)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source=None, addon=None, addition_list:list=None,
                encoding=None, mask=None, kweight=None,
                only_residual=False):
        """
        Args:
            x (torch.Tensor): [N, L, C] (L = im_size/down_factor ** 2)
            source (torch.Tensor): [N, S, C]
            if is_epi_att:
                S = (im_size/down_factor/step_grid) ** 2 * sampling_dim
            else:
                S = im_size/down_factor ** 2
            is_epi_att (bool): Indicates whether it applies epipolar cross-attention
        """
        bs = x.size(0)
        if source is None:
            source = x
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        if addon is not None:
            value = value + addon.view(bs, -1, self.nhead, self.dim).to(value)

        if encoding is not None:
            # shape: [N, L, H, D] -> [N, H, L, D] -> emb -> [N, L, H, D]
            qkhook = lambda x : apply_cached_rotary_emb(encoding, x.transpose(1,2)).transpose(1,2)
        else:
            qkhook = None

        message = self.attention(query, key, value, 
                                 debug=addition_list, mask=mask, kweight=kweight, qkhook=qkhook)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        if only_residual:
            return message
        return x + message
