import torch
import torch.nn as nn
import math
from typing import Optional

# =========================================================
# 1. Scaled Dot-Product Attention (for learning)
# =========================================================
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0
):
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    if dropout > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


# =========================================================
# 2. Multi-Head Attention (from scratch)
# =========================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
   
        B, S, _ = x.shape

        q = self.q_proj(x)  # [B, S, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )

        # Concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)

        # Final projection
        output = self.out_proj(attn_output)
        return output, attn_weights


# =========================================================
# 3. Feed Forward Network
# =========================================================
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 4. Transformer Block (Pre-LN - Modern Standard)
# =========================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pre-LN + Attention + Residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, mask=mask)
        x = x + self.dropout(attn_out)

        # Pre-LN + FFN + Residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# =========================================================
# 5. Simple Causal Mask helper
# =========================================================
def create_causal_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


# =========================================================
# 6. Quick Test
# =========================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    d_model = 128
    num_heads = 4
    d_ff = 512

    x = torch.randn(batch_size, seq_len, d_model)
    causal_mask = create_causal_mask(seq_len)

    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    # Without mask
    out1 = block(x)
    print("Without mask - Output shape:", out1.shape)

    # With causal mask
    out2 = block(x, mask=causal_mask)
    print("With causal mask - Output shape:", out2.shape)

    print("TransformerBlock is working correctly.")