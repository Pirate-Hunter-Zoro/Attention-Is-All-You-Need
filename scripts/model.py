import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.config import BATCH_SIZE, N_EMBD, N_HEAD, BLOCK_SIZE, DROPOUT


class MultiHeadAttention(nn.Module):
    # Single-head self-attention class
    
    def __init__(self):
        super().__init__()
        # Key, query, and value matrices
        if N_EMBD % N_HEAD: # Number of heads does not divide embedding size
            raise ValueError(f"Expected N_HEAD ({N_HEAD}) to evenly divide N_EMBD ({N_EMBD})...")
        self.head_size = N_EMBD // N_HEAD 
        self.key = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.query = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.value = nn.Linear(N_EMBD, N_EMBD, bias=False)
        
        # Upper triangular self attention mask that prevents "reading the future"
        causal_mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)) # Context size x Context size
        self.register_buffer('tril', causal_mask)
        
        self.output_projection = nn.Linear(N_EMBD, N_EMBD) # 'Combines' (throws through a linear layer) concatenated head outputs
        
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x):
        """Input vector
x:                                              (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
after linear projection (via key and query):    (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
after reshape:                                  (BATCH_SIZE, BLOCK_SIZE, N_HEAD, head_size)
after transpose(1,2):                           (BATCH_SIZE, N_HEAD, BLOCK_SIZE, head_size) <-- value(x) after reshaping has these dimensions as well
Q @ K^T:                                        (BATCH_SIZE, N_HEAD, BLOCK_SIZE, BLOCK_SIZE) <-- attention weights, per head
mask + softmax + dropout:                       (BATCH_SIZE, N_HEAD, BLOCK_SIZE, BLOCK_SIZE)
weights @ V:                                    (BATCH_SIZE, N_HEAD, BLOCK_SIZE, head_size)
transpose(1,2) back:                            (BATCH_SIZE, BLOCK_SIZE, N_HEAD, head_size)
reshape back:                                   (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
output projection:                              (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
        """
        if len(x.shape) != 3 or x.shape[0] != BATCH_SIZE or x.shape[1] != BLOCK_SIZE or x.shape[2] != N_EMBD:
            raise ValueError(f"Expected input of shape {(BATCH_SIZE, BLOCK_SIZE, N_EMBD)} but received {x.shape}...")
            
        k = self.key(x)
        k = k.reshape(k.shape[0], k.shape[1], N_HEAD, self.head_size)
        k = k.transpose(1, 2) 
        q = self.query(x) 
        q = q.reshape(q.shape[0], q.shape[1], N_HEAD, self.head_size)
        q = q.transpose(1, 2)
        scaled_attention_weights = torch.matmul(q, k.transpose(-2,-1)) * (self.head_size ** -0.5) 
        masked_weights = scaled_attention_weights.masked_fill(self.tril[:BLOCK_SIZE, :BLOCK_SIZE] == 0, float('-inf'))
        masked_weights = F.softmax(masked_weights, dim=-1)
        masked_weights = self.dropout(masked_weights)
        
        # Now scaled KQ^T attention weight result must be multiplied by value
        v = self.value(x) 
        v = v.reshape(v.shape[0], v.shape[1], N_HEAD, self.head_size)
        v = v.transpose(1, 2)
        masked_weights = torch.matmul(masked_weights, v)
        masked_weights = masked_weights.transpose(1, 2).contiguous().reshape(masked_weights.shape[0], masked_weights.shape[1], N_EMBD)
        
        # Throw through a last linear layer
        return self.dropout(self.output_projection(masked_weights))