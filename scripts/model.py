import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.config import N_LAYER, N_EMBD, N_HEAD, BLOCK_SIZE, DROPOUT


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
        k = self.key(x)
        k = k.reshape(k.shape[0], k.shape[1], N_HEAD, self.head_size)
        k = k.transpose(1, 2) 
        q = self.query(x) 
        q = q.reshape(q.shape[0], q.shape[1], N_HEAD, self.head_size)
        q = q.transpose(1, 2)
        scaled_attention_weights = torch.matmul(q, k.transpose(-2,-1)) * (self.head_size ** -0.5) 
        masked_weights = scaled_attention_weights.masked_fill(self.tril[:x.shape[1], :x.shape[1]] == 0, float('-inf'))
        masked_weights = F.softmax(masked_weights, dim=-1)
        masked_weights = self.dropout(masked_weights)
        
        # Now scaled KQ^T attention weight result must be multiplied by value
        v = self.value(x) 
        v = v.reshape(v.shape[0], v.shape[1], N_HEAD, self.head_size)
        v = v.transpose(1, 2)
        masked_weights = torch.matmul(masked_weights, v)
        masked_weights = masked_weights.transpose(1,2)\
                    .contiguous()\
                    .reshape(masked_weights.shape[0], -1, N_EMBD)
        
        # Throw through a last linear layer
        return self.dropout(self.output_projection(masked_weights))
    

class FeedForward(nn.Module):
    # Follows attention layer in the transformer block
    # "Two linear transformations  with a ReLU in between"
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4*N_EMBD),
            nn.ReLU(),
            nn.Linear(4*N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )
        
    def forward(self, x):
        """
        (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
        ->
        (BATCH_SIZE, BLOCK_SIZE, 4*N_EMBD)
        ->
        (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
        """
        return self.net(x)
    

class Block(nn.Module):
    # Single transformer decoder block
    
    def __init__(self):
        super().__init__()
        self.first_layer_norm = nn.LayerNorm(N_EMBD)
        self.attention_layer = MultiHeadAttention()
        self.second_layer_norm = nn.LayerNorm(N_EMBD)
        self.feed_forward = FeedForward()
        
    def forward(self, x):
        x = x + self.attention_layer(self.first_layer_norm(x))
        x = x + self.feed_forward(self.second_layer_norm(x))
        return x
    
    
class MiniGPT(nn.Module):
    # Wraps all base models together
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        # For each (indexed) word in our vocabulary, map it to a vector
        self.global_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        # For each (indexed) word in our context, map it to a vector
        self.context_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        # Block layers - how many attention heads we have
        self.block_layers = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        # Final layer norm
        self.layer_norm = nn.LayerNorm(N_EMBD)
        # Once we have a final embedded output, we need to turn that into logits (probabilities-esque) over our vocabulary
        self.language_head = nn.Linear(N_EMBD, vocab_size)
        
    def forward(self, x):
        """Receives vector of token indices
        
        Input Embeddigns:
        x: (BATCH_SIZE, BLOCK_SIZE)
        -> Embedding table
        x: (BATCH_SIZE, BLOCK_SIZE, N_EMBD)

        Positional Embeddings:
        indices: (BLOCK_SIZE)
        -> Positional Embedding Table
        indices: (BLOCK_SIZE, N_EMBD)
        """
        embeddings = self.global_embedding_table(x)
        
        token_indices = torch.arange(x.shape[1], device=x.device)
        position_embeddings = self.context_embedding_table(token_indices)
        
        # Broadcasting will handle the batch dimension present in the global embeddings
        positional_embeddings = embeddings + position_embeddings # (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
        
        # Now all of our input tokens have been embedded AND this embedding takes position in context into account
        output_embeddings = self.block_layers(positional_embeddings)
        output_embeddings = self.layer_norm(output_embeddings) # (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
        return self.language_head(output_embeddings) # (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)
    
    def generate(self, x, max_new_tokens):
        """Generate new tokens from the input tokens

        Input tokens:
        x: (BATCH_SIZE, T) -> T is starting sequence length - as we produce new tokens we will only be backing up to BLOCK_SIZE for full context (no farther) to produce the next tokens
        
        """
        # Note we must generate new tokens sequentially because each new token becomes part of the context to generate whatever token follows
        for _ in range(max_new_tokens):
            x = x[:, -BLOCK_SIZE:] # Slice to context size
            # Find logits for next token - only at the last time step
            all_token_logits = self.forward(x)
            next_token_logits = all_token_logits[:, -1, :] # All batches, only last position in context, all token probabilities
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            # For each batch element, sample next tokens according to the element's token probability distribution
            next_tokens = torch.multinomial(next_token_probs, num_samples=1) # (BATCH_SIZE, 1) - next token index for each batch
            x = torch.cat((x, next_tokens), dim=1) # dim 0 is batch, dim 1 is current token indices
        return x # All of the generated tokens - (BATCH_SIZE, T + max_new_tokens)