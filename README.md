# Mini-GPT: Attention Is All You Need

A miniature GPT (decoder-only transformer) built from scratch in PyTorch as a companion project for a midterm presentation on the "Attention Is All You Need" paper. Trained on TinyShakespeare (~1MB) using character-level tokenization to generate Shakespeare-like text.

## Architecture

The model follows the decoder-only transformer architecture with the following components:

| Component | Description |
| --- | --- |
| **MultiHeadAttention** | Fused K/Q/V projections, scaled dot-product attention with causal masking, dropout, and output projection |
| **FeedForward** | Two linear layers (N_EMBD &rarr; 4&times;N_EMBD &rarr; N_EMBD) with ReLU activation and dropout |
| **Block** | Pre-norm transformer block: LayerNorm &rarr; MultiHeadAttention &rarr; residual, then LayerNorm &rarr; FeedForward &rarr; residual |
| **MiniGPT** | Token + positional embeddings, N_LAYER stacked Blocks, final LayerNorm, linear language model head, autoregressive generation via multinomial sampling |

### Hyperparameters

| Parameter | Value |
| --- | --- |
| Batch size | 64 |
| Context length (block size) | 256 |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Transformer layers | 6 |
| Dropout | 0.2 |
| Learning rate | 3e-4 |
| Training iterations | 5,000 |
| Optimizer | AdamW |

## Project Structure

```text
Attention-Is-All-You-Need/
  AttentionIsAllYouNeed.pdf    # Original paper
  Instruction.pdf              # Presentation requirements
  shakespeare.txt              # Training data (TinyShakespeare)
  generated.txt                # Generated output (after training)
  run_training.sbatch          # SLURM job submission script
  scripts/
    config.py                  # Hyperparameters
    model.py                   # Transformer model
    train.py                   # Data loading, training loop, generation
```

## Training

Training uses character-level tokenization (vocabulary size ~65) with a 90/10 sequential train/validation split. Batches are constructed by randomly sampling starting positions within the data and using broadcast indexing to extract sequences in parallel.

Loss is evaluated every 500 iterations by averaging cross-entropy over 200 randomly sampled batches for both train and validation sets. After training, the model generates 5,000 characters of text from a zero-token seed and writes the result to `generated.txt`.

### Running via SLURM

```bash
sbatch run_training.sbatch
```

### Running directly

```bash
python -m scripts.train
```

## Environment Setup

**Server:** NVIDIA A40 (46GB VRAM), CUDA 12.4

### Create conda environment and install dependencies

```bash
conda create -n minigpt python=3.10 -y
conda activate minigpt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install matplotlib
```

### Verify GPU access

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:

```text
True
NVIDIA A40
```

### Important notes

- Do **not** install PyTorch via conda on this server. The `mkl-2025.0.0` package from Anaconda's defaults channel causes `iJIT_NotifyEvent` symbol errors.
- Do **not** mix conda-installed and pip-installed PyTorch in the same environment.
