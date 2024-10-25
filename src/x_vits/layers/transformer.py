import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class AttentionLayer(nn.Module):
    def __init__(self, channels, num_heads, dropout_p):
        super().__init__()
        assert channels % num_heads == 0

        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.rotary_emb = RotaryEmbedding(channels // num_heads, use_xpos=True)
        self.w_q = nn.Linear(channels, channels, bias=False)
        self.w_k = nn.Linear(channels, channels, bias=False)
        self.w_v = nn.Linear(channels, channels, bias=False)
        self.w_o = nn.Linear(channels, channels, bias=False)

    def forward(self, x, attn_mask):
        b, t, c = x.shape

        q = self.w_q(x).reshape(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)
        k = self.w_k(x).reshape(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = self.w_v(x).reshape(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)

        q, k = self.rotary_emb.rotate_queries_and_keys(q, k)
        dropout_p = self.dropout_p if self.training else 0.0
        scores = q @ k.transpose(-2, -1) / c**0.5
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        scores = F.softmax(scores, dim=-1)
        scores = F.dropout(scores, p=dropout_p, training=self.training)
        x = scores @ v
        x = x.transpose(1, 2).contiguous().reshape(b, t, c)
        x = self.w_o(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, channels, context_channels, num_heads, dropout_p):
        super().__init__()
        assert channels % num_heads == 0
        assert context_channels % num_heads == 0

        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.w_q = nn.Linear(channels, channels, bias=False)
        self.w_k = nn.Linear(context_channels, channels, bias=False)
        self.w_v = nn.Linear(context_channels, channels, bias=False)
        self.w_o = nn.Linear(channels, channels, bias=False)

    def forward(self, x, context, attn_mask):
        b, t, c = x.shape
        _, t_c, _ = context.shape

        q = self.w_q(x).reshape(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)
        k = self.w_k(context).reshape(b, t_c, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = self.w_v(context).reshape(b, t_c, self.num_heads, c // self.num_heads).transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0
        scores = q @ k.transpose(-2, -1) / c**0.5
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        scores = F.softmax(scores, dim=-1)
        scores = F.dropout(scores, p=dropout_p, training=self.training)
        x = scores @ v
        x = x.transpose(1, 2).contiguous().reshape(b, t, c)
        x = self.w_o(x)
        return x


# Windowed Relative Positional Encoding is applied
class RelativeMultiHeadAttentionLayer(nn.Module):
    def __init__(self, channels, n_heads, dropout, window_size=4):
        super().__init__()
        assert channels % n_heads == 0

        self.inter_channels = channels // n_heads
        self.n_heads = n_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.inter_channels)

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

        rel_stddev = self.inter_channels**-0.5
        self.emb_rel_k = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.inter_channels) * rel_stddev
        )
        self.emb_rel_v = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.inter_channels) * rel_stddev
        )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, attn_mask):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        B, C, T = q.size()
        query = q.view(B, self.n_heads, self.inter_channels, T).transpose(2, 3)
        key = k.view(B, self.n_heads, self.inter_channels, T).transpose(2, 3)
        value = v.view(B, self.n_heads, self.inter_channels, T).transpose(2, 3)

        scores = torch.matmul(query / self.scale, key.transpose(-2, -1))

        pad_length = max(0, T - (self.window_size + 1))
        start = max(0, (self.window_size + 1) - T)
        end = start + 2 * T - 1

        pad_rel_emb = F.pad(self.emb_rel_k, [0, 0, pad_length, pad_length, 0, 0])
        k_emb = pad_rel_emb[:, start:end]

        rel_logits = torch.matmul(query / self.scale, k_emb.unsqueeze(0).transpose(-2, -1))
        rel_logits = F.pad(rel_logits, [0, 1])
        rel_logits = rel_logits.view([B, self.n_heads, 2 * T * T])
        rel_logits = F.pad(rel_logits, [0, T - 1])
        scores_local = rel_logits.view([B, self.n_heads, T + 1, 2 * T - 1])[:, :, :T, T - 1 :]

        scores = scores + scores_local
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        p_attn = F.pad(p_attn, [0, T - 1])
        p_attn = p_attn.view([B, self.n_heads, T * (2 * T - 1)])
        p_attn = F.pad(p_attn, [T, 0])
        relative_weights = p_attn.view([B, self.n_heads, T, 2 * T])[:, :, :, 1:]

        pad_rel_emb = F.pad(self.emb_rel_v, [0, 0, pad_length, pad_length, 0, 0])
        v_emb = pad_rel_emb[:, start:end]

        output = output + torch.matmul(relative_weights, v_emb.unsqueeze(0))

        x = output.transpose(2, 3).contiguous().view(B, C, T)

        x = self.conv_o(x)
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, channels, dropout_p, scale=4):
        super().__init__()

        self.channels = channels
        self.w1 = nn.Linear(channels, channels * scale)
        self.w2 = nn.Linear(channels, channels * scale)
        self.w3 = nn.Linear(channels * scale, channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.dropout(x)
        x = self.w3(x)
        return x * mask


class VITSFeedForwardLayer(FeedForwardLayer):
    def __init__(self, channels, dropout_p, scale=4):
        super().__init__(channels, dropout_p, scale=scale)

        self.channels = channels
        self.w1 = nn.Conv1d(channels, channels * scale, 3, padding=1)
        self.w2 = nn.Conv1d(channels, channels * scale, 3, padding=1)
        self.w3 = nn.Conv1d(channels * scale, channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout_p)
