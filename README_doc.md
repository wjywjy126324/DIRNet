# DIRNet: Prompt Learning-Driven Unified Image Restoration

## Project Overview

This project implements **DIRNet**, a prompt-learning-guided all-in-one image restoration network introduced in the paper *"Prompt Learning-Driven Unified Image Restoration for Industrial Silicon Nitride Ceramic Ball Surface Inspection"* (published in Journal of Intelligent Manufacturing, 2026). The model is built upon the PromptIR framework and introduces a **Hybrid Attention Mechanism based on Visual Prompts (HAM-VP)** and a **Lightweight Transformer Module** to explicitly decouple degradation responses from structural semantics, enabling unified restoration of multiple degradation types (low-brightness, blur, noise) in a single model.

---

## 1. Dependencies and Requirements

### 1.1 Hardware Environment

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 3070 Ti (recommended) |
| CUDA | 11.6 |
| RAM | 16 GB+ |
| OS | Linux / Windows |

### 1.2 Software Dependencies

#### Core Deep Learning Framework

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.8+ | Tested with 3.8.11 |
| PyTorch | 1.8.1+ | CUDA-enabled build required |
| Torchvision | 0.9.1+ | Compatible with PyTorch version |
| Torchaudio | 0.8.1+ | Optional, for audio-related extensions |

#### Training & Experiment Utilities

| Package | Version | Notes |
|---------|---------|-------|
| pytorch-lightning | 2.0.1 | Used in `test.py` for model checkpointing and LightningModule wrapper |
| lightning | 2.0.1 | Core Lightning framework |
| accelerate | 0.18.0 | Distributed training acceleration |
| deepspeed | 0.8.3 | Optional distributed training |
| warmup-scheduler | 0.3.2 | Custom LR warmup support |

#### Scientific Computing & Data Processing

| Package | Version | Notes |
|---------|---------|-------|
| NumPy | 1.20.3+ | Array operations |
| SciPy | 1.6.2+ | Scientific computing |
| scikit-image | 0.19.3+ | Image processing utilities |
| scikit-learn | 1.0.1+ | ML utilities |
| pandas | 1.5.3+ | Tabular data handling |
| opencv-python | 4.7.0.68 | Image I/O and transformations |
| Pillow (PIL) | 9.4.0+ | Image loading and augmentation |
| imageio | 2.25.0+ | Image I/O |
| pywavelets | 1.4.1+ | Wavelet transforms (optional) |

#### Attention & Transformer Support

| Package | Version | Notes |
|---------|---------|-------|
| einops | 0.6.0 | Elegant tensor reshaping (e.g., `rearrange`) |
| mmcv / mmcv-full | 1.7.1 | Computer vision utilities |
| timm | 0.6.12 | Pretrained model utilities |

#### Logging & Visualization

| Package | Version | Notes |
|---------|---------|-------|
| wandb | 0.13.9 | Experiment tracking (optional) |
| matplotlib | 3.6.3+ | Plotting |
| seaborn | 0.12.2+ | Statistical visualization |

#### Other Utilities

| Package | Version | Notes |
|---------|---------|-------|
| tqdm | 4.62.0+ | Progress bars |
| PyYAML | 6.0 | Configuration file parsing |
| addict | 2.4.0 | Dictdot access |
| packaging | 23.1 | Version parsing |
| requests | 2.28.2 | HTTP requests |

### 1.3 Quick Setup via `env.yml`

The project includes a complete conda environment specification in `env.yml`. To reproduce the environment:

```bash
conda env create -f env.yml
conda activate promptir
```

Or install via pip (selected key packages):

```bash
pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1
pip install einops==0.6.0 pytorch-lightning==2.0.1 lightning==2.0.1
pip install mmcv==1.7.1 mmcv-full==1.7.1 timm==0.6.12
pip install opencv-python==4.7.0.68 scikit-image==0.19.3
pip install wandb==0.13.9
```

### 1.4 Training Configuration (from `options.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Total training epochs |
| `--batch_size` | 1 | Batch size per GPU |
| `--lr` | 2e-4 | Initial learning rate |
| `--patch_size` | 128 | Input patch resolution |
| `--num_workers` | 0 | DataLoader worker processes |
| `--cuda` | 0 | GPU device ID |
| `--de_type` | denoise_15/25/50, derain, dehaze | Degradation types for training |
| `--ckpt_dir` | "" | Checkpoint save directory |
| `--data_file_dir` | "" | Root path for training data |
| `--distributed` | False | Enable distributed training |

---

## 2. Key Algorithms Description and Implementation

### 2.1 Overall Architecture: PromptIR Backbone with DIRNet Enhancements

DIRNet adopts a **four-level U-shaped encoder-decoder structure** built on top of the PromptIR architecture. The model progressively extracts multi-scale features from a degraded input image and reconstructs the clean image through a symmetric decoder with skip connections.

**Core architectural parameters:**

| Component | Specification |
|-----------|--------------|
| Input channels | 3 (RGB) |
| Output channels | 3 (RGB) |
| Base feature dimension | 48 |
| Transformer blocks per level | [4, 6, 6, 8] (L1–L4) |
| Attention heads per level | [1, 2, 4, 8] |
| FFN expansion factor | 2.66 |
| Layer normalization type | WithBias (optional: BiasFree) |
| Refinement blocks | 4 |
| Decoder prompt modules | Enabled (`decoder=True`) |

**Feature progression:**
```
Level 1: C×H×W    →  Encoder → Decoder → Output
Level 2: 2C×H/2×W/2
Level 3: 4C×H/4×W/4
Level 4: 8C×H/8×W/8  (Latent space)
```

The decoder uses **Pixel Shuffle** (up-sampling) symmetrically to the encoder's **Pixel Unshuffle** (down-sampling). Final output is computed via residual learning: `I_out = I_in + F_R`.

### 2.2 Lightweight Transformer Block

Each level of the encoder/decoder consists of lightweight Transformer blocks, each containing two sub-units: **WMDTA** (Window Multi-DConv Head Transposed Attention) and **NLGDFN** (Non-Linear Gated-Dconv Feed-Forward Network).

#### 2.2.1 WMDTA — Window Multi-DConv Head Transposed Self-Attention

**Motivation:** Standard self-attention has O((H×W)²) complexity, which is prohibitive for high-resolution industrial images. WMDTA reduces this by restricting attention to local windows and replacing matrix multiplication with depthwise convolutions.

**Implementation (`model.py`, `Win_MDTA.py`):**

1. **Local convolution preprocessing**: A 3×3 depthwise convolution extracts local context before attention computation.
2. **QKV projection**: 1×1 convolutions generate query, key, and value tensors.
3. **L2 normalization**: Q and K are normalized before dot-product to stabilize training.
4. **Window-based attention**: The input is partitioned into windows of size `window_size` (default 7×7), and self-attention is computed independently within each window.
5. **Gated output**: The attention output is projected back via 1×1 convolution and added to the input residually.

**Complexity:** Reduced from O(N²) to approximately O(window_size²) per window, making it tractable for high-resolution inputs.

#### 2.2.2 NLGDFN — Non-Linear Gated-Dconv Feed-Forward Network

**Motivation:** Traditional FFNs perform only linear mapping and nonlinear transformation. NLGDFN enhances information flow modulation by introducing a three-branch gating mechanism with L2 normalization and Mish activation.

**Implementation (`model.py`):**

1. **Channel expansion**: Input is projected to `dim × ffn_expansion_factor` channels via 1×1 convolution.
2. **Depth convolution**: 3×3 depthwise convolution captures spatial context in each branch.
3. **Three-branch gating**: Two branches generate spatial attention weights via L2 normalization and Mish activation; the third branch carries the content. Gates are applied element-wise.
4. **Channel reduction**: 1×1 convolution maps back to the original channel dimension.
5. **Mish activation**: Used throughout instead of GELU or ReLU for smoother gradient flow.

### 2.3 HAM-VP — Hybrid Attention Mechanism based on Visual Prompts

HAM-VP is the core feature modulation module that addresses the degradation-structure entanglement problem. It consists of four sequential units deployed at encoder-decoder residual connections.

#### 2.3.1 SDFF — Spatial Domain Feature Fusion

Splits features into local and global branches:
- **Local branch**: Group normalization + weighted gating, followed by channel shuffle and cross-recombination to preserve fine-grained textures.
- **Global branch**: Average pooling across height and width → 4-group channel split → multi-scale 1D depthwise convolutions (kernels: 3, 5, 7, 9) → Sigmoid gating.
- **Fusion**: Element-wise multiplication of local and global outputs.

#### 2.3.2 CDFF — Channel Domain Feature Fusion

Processes channel statistics for degradation pattern discrimination:
- **Upper branch**: Channel-wise group splitting → parallel paths (group conv + pointwise conv vs. pointwise only) → Sigmoid gating → recalibration.
- **Lower branch**: Global average pooling → GroupNorm → 3 parallel depthwise convolutions for Q/K/V → single-head self-attention → gating.
- **Fusion**: Element-wise multiplication of both branch outputs.

#### 2.3.3 RHFF — Redundant Hybrid Feature Filtering

Filters out noise-related redundant responses using gating:
1. Layer normalization → 1×1 point convolution → linear projection.
2. Split into two parts: one as benchmark, one as gating signal.
3. The gating branch goes through reshape → depthwise conv → flatten → gating multiplication with the benchmark.
4. Residual connection to the original input.

#### 2.3.4 APHF — Alignment of Prompt Features with Hybrid Features

Aligns prompt block outputs with mixed features for explicit degradation guidance:
1. Dynamic dimension alignment via linear interpolation (spatial) and truncation/padding (channel).
2. Element-wise multiplication with the prompt base.
3. Concatenation with prompt block output → 1×1 convolution fusion.

### 2.4 Prompt Generation Module

The `PromptGenBlock` (`model.py`) generates degradation-aware prompts dynamically from input features:

```
Input features (B×C×H×W)
    → Global average pooling (B×C)
    → Linear projection → Softmax → Prompt weights
    → Weighted sum of learnable prompt tensors
    → Bilinear interpolation to match spatial resolution
    → 3×3 convolution
    → Output prompt (B×prompt_dim×H×W)
```

Three prompt generators are deployed at different decoder levels (channels: 64, 128, 320), each with prompt length 5 and learnable parameters of shape `1×5×prompt_dim×prompt_size×prompt_size`.

### 2.5 SCSA — Spatial-Channel Self-Attention Module

SCSA (`My_module.py`) combines spatial and channel attention with redundant feature filtering:

**Spatial path:**
1. **SRU (Spatial Reconstruction Unit)**: GroupNorm → Sigmoid gating with learnable weights → reconstructive split-and-add.
2. **Multi-scale spatial attention**: Average pooling across H and W separately → 4-group channel split (each with different 1D depthwise kernel sizes: 3, 5, 7, 9) → Sigmoid-weighted attention maps applied along H and W dimensions.

**Channel path:**
1. **Down-sampling**: AvgPool (window_size×window_size) to reduce spatial resolution.
2. **Self-attention**: Grouped QKV (1 head) → dot-product attention → channel-wise gating.
3. **CRU (Channel Reconstruction Unit)**: Channel split → squeeze-and-excitation with GWC (grouped weighted conv) + PWC (pointwise conv) → adaptive average pooling → softmax fusion → split-and-add.

### 2.6 FRFN — Fast Residual Feed-Forward Network

FRFN (`FRFN.py`) provides efficient spatial-channel feature transformation with a gating mechanism:

```
Input (B×HW×C)
    → Reshape to (B×C×H×W)
    → Partial depthwise conv (3×3) on C/4 channels
    → Concatenate with untouched C×3/4 channels
    → Flatten back to (B×HW×C)
    → Linear projection → Split into two halves
    → Reshape first half → 3×3 depthwise conv → Mish activation
    → Flatten → Gate multiplication with second half
    → Linear projection → Output
```

### 2.7 ScConv — Spatial & Channel Reconstruction Unit

ScConv (`ScConv.py`) is a composite unit combining SRU and CRU:

- **SRU**: Splits channels into two halves via a gating threshold on normalized features; reconstructs via cross-half addition.
- **CRU**: Splits channels by ratio α → upper path uses grouped weighted conv + pointwise conv; lower path uses pointwise conv → concat → adaptive softmax pooling → split-and-add.

### 2.8 Training Strategy

| Item | Details |
|------|---------|
| Loss function | L1 Loss (`nn.L1Loss`) |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| LR schedule | Linear Warmup (15 epochs) + Cosine Annealing |
| Input preprocessing | Random 128×128 crops, random horizontal/vertical flips |
| Degradation simulation | Gaussian noise (σ ∈ {15, 25, 50}), real derain/dehaze datasets |
| Checkpointing | Every epoch; saves full model + optimizer state |

**Ablation findings from the paper:**
- HAM-VP alone improves Avg.PSNR by ~0.48 dB and Avg.SSIM by 0.011 over the PromptIR baseline.
- The lightweight Transformer module improves Avg.PSNR by ~0.23 dB with comparable computational cost.
- Full DIRNet achieves +4.38 dB over the baseline on SINCB-DID, reaching 34.73 dB / 0.910.

### 2.9 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB), higher is better |
| **SSIM** | Structural Similarity Index [0,1], higher is better |
| **FPS** | Frames per second (throughput) |
| **Avg. PSNR/SSIM** | Average across all degradation types |

---

## 3. Project Structure

```
PromptIR-main/
├── model.py               # Core PromptIR model + PromptGenBlock + TransformerBlock + Attention
├── FRFN.py               # Fast Residual Feed-Forward Network
├── Win_MDTA.py           # Window Multi-DConv Head Transposed Self-Attention
├── My_module.py          # SCSA (Spatial-Channel Self-Attention) module
├── ScConv.py             # SRU + CRU composite unit
├── Torch/
│   └── mish.py          # Mish activation function
├── train.py              # Training script (vanilla PyTorch)
├── test.py               # Testing script (PyTorch Lightning wrapper)
├── options.py            # CLI argument definitions and configuration namespace
├── env.yml               # Conda environment specification
└── utils/
    ├── dataset_utils.py   # PromptTrainDataset, DenoiseTestDataset, DerainDehazeDataset
    ├── degradation_utils.py  # Gaussian noise degradation simulation
    ├── schedulers.py      # LR schedulers (CosineAnnealing, Warmup, etc.)
    ├── val_utils.py       # PSNR/SSIM computation utilities
    ├── image_io.py        # Image save/load utilities
    ├── image_utils.py     # Crop, augmentation helpers
    ├── loss_utils.py      # Loss function utilities
    ├── imresize.py        # Image resizing utilities
    └── pytorch_ssim/      # SSIM implementation
```

---

## 4. Key Implementation Details

### 4.1 Custom Layer Normalizations

Two variants are provided in `model.py`:
- **BiasFree_LayerNorm**: Only learnable scale (weight), no bias. Used when avoiding channel bias is desired.
- **WithBias_LayerNorm**: Standard layer norm with both scale and bias parameters.

Both accept input in `(B, C, H, W)` format and internally reshape to `(B, H×W, C)` for normalization.

### 4.2 Down/Upsampling

- **Downsample**: `PixelUnshuffle(2)` + 3×3 convolution (reduces H, W by 2×, doubles C).
- **Upsample**: 3×3 convolution + `PixelShuffle(2)` (increases H, W by 2×, halves C).

### 4.3 SCSA Integration in PromptIR

SCSA modules are inserted at encoder output skip connections at all three levels:
```python
self.my_module3 = SCSA(192)  # Level 3
self.my_module2 = SCSA(96)   # Level 2
self.my_module1 = SCSA(48)  # Level 1
```

### 4.4 FRFN Integration

FRFN modules are similarly placed at skip connections (currently commented out in the forward pass, available for activation):
```python
self.frfn3 = FRFN(192)
self.frfn2 = FRFN(96)
self.frfn1 = FRFN(48)
```

### 4.5 Mixed Degradation Training

The data pipeline supports training on multiple degradation types simultaneously:
- Gaussian denoising: σ = 15, 25, 50 (synthetically added)
- Deraining: rain streak removal (real paired data)
- Dehazing: single-image dehazing (real paired data)

The `PromptTrainDataset` merges samples from all specified degradation types with resampling to balance class frequencies.

---

## 5. Performance Summary

### 5.1 Results on SINCB-DID Dataset

| Degradation | Metric | DIRNet | PromptIR | Best Competitor |
|-------------|--------|---------|----------|-----------------|
| Low-Brightness | PSNR | 21.03 dB | 19.77 dB | 19.77 dB |
| Deblurring | PSNR | 34.49 dB | 33.55 dB | 33.94 dB |
| Denoising σ=15 | PSNR | 40.72 dB | 40.07 dB | 39.86 dB |
| Denoising σ=25 | PSNR | 39.58 dB | 38.66 dB | 38.45 dB |
| Denoising σ=50 | PSNR | 37.83 dB | 35.07 dB | 34.77 dB |
| **Average** | PSNR/SSIM | **34.73/0.910** | 33.42/0.887 | 32.55/0.907 |

### 5.2 Generalization on Public Datasets

| Dataset | Task | DIRNet PSNR/SSIM |
|---------|------|-----------------|
| LOL | Low-light enhancement | 20.93/0.823 |
| GoPro | Image deblurring | 28.14/0.834 |
| BSD68 | Image denoising | 32.56/0.902 |

### 5.3 Module Complexity Comparison

| Module | Parameters | FLOPs |
|--------|-----------|-------|
| PIP (baseline prompt module) | 0.7 M | 15.580 G |
| **HAM-VP** | **0.1 M** | **0.681 G** |
| Standard Transformer Block | 140 K | 0.205 G |
| **Lightweight Transformer Block** | **150 K** | **0.202 G** |

---

## 6. Usage

### 6.1 Training

```bash
python train.py \
    --data_file_dir /path/to/data \
    --denoise_dir /path/to/denoise \
    --ckpt_dir /path/to/checkpoints \
    --epochs 120 \
    --batch_size 1 \
    --lr 2e-4 \
    --patch_size 128 \
    --de_type denoise_15 denoise_25 denoise_50 derain dehaze
```

### 6.2 Testing

```bash
python test.py \
    --mode 3 \
    --ckpt_name /path/to/checkpoint.ckpt \
    --denoise_path /path/to/test/denoise \
    --derain_path /path/to/test/derain \
    --dehaze_path /path/to/test/dehaze \
    --output_path /path/to/output \
    --cuda 0
```

### 6.3 Notes

- `test.py` uses PyTorch Lightning's `PromptIRModel` wrapper to load `.ckpt` checkpoints. Ensure the checkpoint format matches.
- `train.py` uses vanilla PyTorch and saves `.pth` checkpoints. The two formats are not directly interchangeable without adaptation.
- Window size for WMDTA is 7×7 by default. Adjust via the `window_size` parameter in `Attention`.
- Set `decoder=True` in `PromptIR(...)` to enable prompt generation modules.
