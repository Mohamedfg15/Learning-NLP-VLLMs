import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional

# =========================================================
# 1. Core Components 
# =========================================================

def scaled_dot_product_attention(query, key, value, mask=None, dropout=0.0):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    if dropout > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


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

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(attn_output)


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


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


def create_causal_mask(seq_len: int, device="cpu"):
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


# =========================================================
# 2. Mini Language Model
# =========================================================

class MiniTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.max_seq_len = max_seq_len

    def forward(self, idx):
       
        B, S = idx.shape
        device = idx.device

        tok_emb = self.token_emb(idx)
        pos = torch.arange(S, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)

        x = self.dropout(tok_emb + pos_emb)

        
        mask = create_causal_mask(S, device=device)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)
        logits = self.lm_head(x)          
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# =========================================================
# 3. Simple Character-level Dataset
# =========================================================

class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 32):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.seq_len = seq_len

        data = [self.stoi[c] for c in text]
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# =========================================================
# 4. Training + Generation Example
# =========================================================

if __name__ == "__main__":
   
    text = """
    computer vision is a field of artificial intelligence.
    deep learning powers modern computer vision systems.
    transformers changed natural language processing forever.
    attention is all you need.
    machine learning models learn patterns from data.
    """ * 20

    seq_len = 32
    batch_size = 16
    d_model = 128
    num_heads = 4
    num_layers = 4

    dataset = CharDataset(text, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniTransformerLM(
        vocab_size=dataset.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=512,
        max_seq_len=seq_len
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Vocab size: {dataset.vocab_size}")
    print("Starting training...\n")

    
    model.train()
    for epoch in range(1, 11):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")

    
    model.eval()
    start_text = "computer vision"
    start_ids = torch.tensor([[dataset.stoi[c] for c in start_text]], dtype=torch.long).to(device)

    generated = model.generate(start_ids, max_new_tokens=80, temperature=0.8)
    generated_text = "".join([dataset.itos[i.item()] for i in generated[0]])

    print("\n===== Generated Text =====")
    print(generated_text)