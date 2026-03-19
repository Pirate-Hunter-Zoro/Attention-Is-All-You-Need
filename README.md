# Mini-GPT: Attention Is All You Need

## Environment Setup

**Server:** compute303 (NVIDIA A40, 46GB VRAM, CUDA 13.1 driver)

### Step 1: Create conda environment (Python only)

```bash
conda create -n minigpt python=3.10 -y
conda activate minigpt
```

### Step 2: Install PyTorch via pip (NOT conda)

Conda's `mkl-2025.0.0` package from the defaults channel is broken on this system
(`undefined symbol: iJIT_NotifyEvent` in `libtorch_cpu.so`). PyTorch's pip wheels
bundle their own MKL and CUDA runtime, avoiding the conflict entirely.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 3: Install additional packages

```bash
pip install tiktoken matplotlib
```

### Step 4: Verify

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:

```text
True
NVIDIA A40
```

### Installed versions

- Python 3.10
- PyTorch 2.6.0+cu124
- CUDA toolkit 12.4 (bundled with pip wheels)

### What NOT to do

- Do NOT install PyTorch via conda on this server. The `mkl-2025.0.0` package
  from Anaconda's defaults channel causes `iJIT_NotifyEvent` symbol errors.
- Do NOT mix conda-installed and pip-installed PyTorch in the same environment.
