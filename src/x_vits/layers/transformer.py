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

        q, k = self.rotary_emb.rotate_queries_and_keys(q, k, seq_dim=2)
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
