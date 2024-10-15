import torch.nn as nn

from x_vits.layers import AdaRMSNorm, AttentionLayer, CrossAttentionLayer, FeedForwardLayer


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, dropout_p, context_channels=0):
        super().__init__()

        self.attn_norm = nn.LayerNorm(channels)
        self.attn = AttentionLayer(channels, num_heads, dropout_p)
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = FeedForwardLayer(channels, dropout_p)

        self.with_cross = context_channels > 0
        if self.with_cross:
            self.xattn_norm = nn.LayerNorm(channels)
            self.xattn = CrossAttentionLayer(channels, context_channels, num_heads, dropout_p)
            self.xff_norm = nn.LayerNorm(channels)
            self.xff = FeedForwardLayer(channels, dropout_p)

    def forward(self, x, x_mask, attn_mask, context=None, context_attn_mask=None):
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + self.ff(self.ff_norm(x), x_mask)
        if self.with_cross:
            x = x + self.xattn(self.xattn_norm(x), context=context, attn_mask=context_attn_mask)
            x = x + self.xff(self.xff_norm(x), x_mask)
        return x


class AdaTransformerBlock(TransformerBlock):
    def __init__(self, channels, cond_channels, num_heads, dropout_p, context_channels=0):
        super().__init__()

        self.attn_norm = AdaRMSNorm(channels, cond_channels)
        self.attn = AttentionLayer(channels, num_heads, dropout_p)
        self.ff_norm = AdaRMSNorm(channels, cond_channels)
        self.ff = FeedForwardLayer(channels, dropout_p)
        self.gate = nn.Linear(cond_channels, channels, bias=False)

        self.with_cross = context_channels > 0
        if self.with_cross:
            self.xattn_norm = AdaRMSNorm(channels, cond_channels)
            self.xattn = CrossAttentionLayer(channels, context_channels, num_heads, dropout_p)
            self.xff_norm = AdaRMSNorm(channels, cond_channels)
            self.xff = FeedForwardLayer(channels, dropout_p)
            self.xgate = nn.Linear(cond_channels, channels, bias=False)

    def forward(self, x, x_mask, attn_mask, cond, context=None, context_attn_mask=None):
        gate = self.scale(cond)[:, None]
        x = x + self.attn(self.attn_norm(x, cond=cond), attn_mask=attn_mask)
        x = x + gate * self.ff(self.ff_norm(x, cond=cond), x_mask)
        if self.with_cross:
            x_gate = self.xgate(cond)[:, None]
            x = x + self.xattn(
                self.xattn_norm(x, cond=cond), context=context, attn_mask=context_attn_mask
            )
            x = x + x_gate * self.xff(self.xff_norm(x, cond=cond), x_mask)
        return x
