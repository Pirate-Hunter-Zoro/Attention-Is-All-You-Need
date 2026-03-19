import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.config import N_EMBD, BLOCK_SIZE, DROPOUT

class Head(nn.Module):
    # Single-head self-attention class
    
    def __init__(self, head_size: int):
        super().__init__()
        # Key, query, and value matrices
        self.head_size = head_size
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        
        # Upper triangular self attention mask that prevents "reading the future"
        causal_mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)) # Context size x Context size
        self.register_buffer('tril', causal_mask)
        
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x):
        _, T, _ = x.shape
        # x is of shape (B, T, C), which is batch size, context size, embedding size
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Compute attention weights - for each batch, we have a T x T (context size) matrix where each token in the context sequence has an attention value for other tokens
        scaled_attention_weights = torch.matmul(q, k.transpose(-2,-1)) * (self.head_size ** -0.5) # B x T x T
        # But don't let current tokens pay attention to future tokens
        masked_weights = scaled_attention_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        masked_weights = F.softmax(masked_weights, dim=-1)
        masked_weights = self.dropout(masked_weights)
        v = self.value(x) # B x T x head_size
        return torch.matmul(masked_weights, v) # B x T x head_size