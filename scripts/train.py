import torch
from pathlib import Path
import torch.nn.functional as F

from scripts.config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    LEARNING_RATE,
    MAX_ITERS,
    EVAL_INTERVAL,
    EVAL_ITERS
)
from scripts.model import MiniGPT

def get_batch(data):
    # Pick random starting positions
    starts = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    offsets = torch.arange(BLOCK_SIZE) # (BLOCK_SIZE,) -> 1,2,3,...BLOCK_SIZE
    batches_x_indices = starts.unsqueeze(1) + offsets # (BATCH_SIZE, BLOCK_SIZE)
    batches_y_indices = (starts + 1).unsqueeze(1) + offsets
    x = data[batches_x_indices].to(DEVICE)
    y = data[batches_y_indices].to(DEVICE) # Target from this sequence is the very next sequence after progressing one character
    # Now we have our random-start sequences of tokens (each represented with an index)
    return x, y

def estimate_loss(model, train_data, val_data):
    # No gradient tracking here - this is just for monitoring
    with torch.no_grad():
        model.eval() # Dropout and BatchNorm are no longer stochastic until we call no_grad()
        out = {"train_loss": 0.0, "test_loss": 0.0}
        for _ in range(EVAL_ITERS):
            x_train, y_train = get_batch(train_data)
            y_train_pred = model(x_train)
            # Flatten over all batches, and sequences in each batch, to get a long sequence of tokens - flatten the target output token index sequence as well
            train_loss = F.cross_entropy(y_train_pred.view(-1, model.vocab_size), y_train.view(-1))
            out["train_loss"] += train_loss.item()
            
            x_test, y_test = get_batch(val_data)
            y_test_pred = model(x_test)
            test_loss = F.cross_entropy(y_test_pred.view(-1, model.vocab_size), y_test.view(-1))
            out["test_loss"] += test_loss.item()
        
        model.train()
        # We want average loss over the iterations
        out["train_loss"] /= EVAL_ITERS
        out["test_loss"] /= EVAL_ITERS
        return out

def main():
    all_text = (Path(__file__).resolve().parent.parent / "shakespeare.txt").read_text()
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    
    # Turn text into token IDs
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)} # For decoding later
    # Turn every character into an index
    text_token_ids = torch.tensor([stoi[ch] for ch in all_text], dtype=torch.long)
    # Split the tokens into training and validation sets
    n = int(0.9 * len(text_token_ids))
    train_data = text_token_ids[:n]
    val_data = text_token_ids[n:]
    print(f"Found {len(text_token_ids)} total tokens", flush=True)
    
    # Training loop
    model = MiniGPT(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for i in range(MAX_ITERS):
        x, y = get_batch(train_data)
        # Prediction token logits
        logits = model(x)
        # Just like in the above evaluation function, we flatten over the batches before computing loss
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % EVAL_INTERVAL == 0:
            out = estimate_loss(model, train_data, val_data)
            print(f"Epoch {i+1}:\n\
Training Loss: {out['train_loss']}\n\
Testing Loss: {out['test_loss']}\n", flush=True)
            
    # Now that the model has trained, let us generate some text
    model.eval()
    # Start with a random token - single batch, single token - for our seed
    x = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
    output_logits = model.generate(x, max_new_tokens=50000).squeeze().tolist()
    result = "".join([itos[idx] for idx in output_logits])
    with open((Path(__file__).resolve().parent.parent / "generated.txt"), 'w') as f:
        f.write(result)
    
if __name__=="__main__":
    main()