import torch

BATCH_SIZE = 64 # Number of independent sequences to process in parallel
BLOCK_SIZE = 256 # Maximum context length
N_EMBD = 384 # Embedding dimension of the model
N_HEAD = 6 # Number of attention heads - must evenly divide the embedding dimension
N_LAYER = 6 # Number of stacked transformer blocks
DROPOUT = 0.2 # Dropout rate for training
LEARNING_RATE = 3e-4 # Learning rate for training
MAX_ITERS = 5000 # Training iterations
EVAL_INTERVAL = 500 # How often loss is evaluated
EVAL_ITERS = 200 # Number of batches to average over when calculating loss
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Cuda if available