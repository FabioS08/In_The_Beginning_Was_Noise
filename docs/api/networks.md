# Networks API

:material-file-code: `models/networks/`

---

## UNet

:material-file-code: `models/networks/unet.py`

The U-Net architecture for diffusion models. Built dynamically from configuration lists.

```python
class UNet(nn.Module):
    def __init__(
        self,
        timeStepEmbedding: TimeStepEmbedding,
        initChannels: int,
        encoder: List[EncoderItem],
        bottleneck: List[BottleneckItem],
        decoder: List[DecoderItem]
    )
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `timeStepEmbedding` | `TimeStepEmbedding` | Timestep embedding module |
| `initChannels` | `int` | Output channels of the entry convolution |
| `encoder` | `List[EncoderItem]` | Encoder block specifications |
| `bottleneck` | `List[BottleneckItem]` | Bottleneck block specifications |
| `decoder` | `List[DecoderItem]` | Decoder block specifications |

### `forward(x, t) → Tensor`

| Input | Shape | Description |
|:------|:------|:------------|
| `x` | `[B, 3, H, W]` | Noisy input image |
| `t` | `[B]` | Integer timestep indices |

| Output | Shape | Description |
|:-------|:------|:------------|
| `out` | `[B, 3, H, W]` | Predicted noise |

### Internal Components

| Component | Description |
|:----------|:------------|
| `entryConv` | `Conv2d(3 → initChannels, 3×3)` |
| `encoder` | `ModuleList` of encoder blocks |
| `bottleneck` | `ModuleList` of bottleneck blocks |
| `decoder` | `ModuleList` of decoder blocks |
| `exit` | `GroupNorm → SiLU → Conv2d(C → 3, 3×3)` |

### Exceptions

#### `SkipConnectionsError`

Raised when skip connections are misaligned:

- **Empty stack**: Decoder tries to pop from an empty skip stack
- **Non-empty stack**: Skip stack is not empty after the decoder finishes

---

## TimeStepEmbedding

:material-file-code: `models/networks/embeddings.py`

Sinusoidal positional encoding followed by a learnable MLP projection.

```python
class TimeStepEmbedding(nn.Module):
    def __init__(self, embeddingDimension: int)
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `embeddingDimension` | `int` | Base sinusoidal dimension (**must be even**) |

| Attribute | Value | Description |
|:----------|:------|:------------|
| `outputDimension` | `4 × embeddingDimension` | Final output dimension after MLP |

### `forward(t) → Tensor`

| Input | Shape | Description |
|:------|:------|:------------|
| `t` | `[B]` | Integer timesteps |

| Output | Shape | Description |
|:-------|:------|:------------|
| `embedding` | `[B, 4 × embeddingDimension]` | Projected timestep embedding |

### Pipeline

```
t: [B] → Sinusoidal PE: [B, d] → Linear(d, 4d) → SiLU → Linear(4d, 4d) → [B, 4d]
```

---

## Building Blocks

:material-file-code: `models/networks/blocks/`

### ResBlock

:material-file-code: `blocks/resblock.py`

Residual convolutional block with timestep conditioning.

```python
class ResBlock(nn.Module):
    def __init__(self, inChannels, outChannels, dTimeEmbedding, numGroups=32, dropout=0.1)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `inChannels` | `int` | — | Input channels |
| `outChannels` | `int` | — | Output channels |
| `dTimeEmbedding` | `int` | — | Timestep embedding dimension |
| `numGroups` | `int` | `32` | GroupNorm groups |
| `dropout` | `float` | `0.1` | Dropout probability |

**Input:** `x: [B, C_in, H, W]`, `t: [B, d_time]`  
**Output:** `[B, C_out, H, W]`

!!! warning "GroupNorm constraint"
    Both `inChannels` and `outChannels` must be divisible by `numGroups`. A `ValueError` is raised otherwise.

---

### MultiHeadAttentionBlock

:material-file-code: `blocks/attention.py`

Spatial multi-head self-attention with residual connection.

```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, numHeads, inChannels, numGroups=32, zeroInit=True)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `numHeads` | `int` | — | Number of attention heads |
| `inChannels` | `int` | — | Input channels |
| `numGroups` | `int` | `32` | GroupNorm groups |
| `zeroInit` | `bool` | `True` | Zero-initialize output projection |

**Input:** `x: [B, C, H, W]`  
**Output:** `[B, C, H, W]`

Uses `torch.nn.functional.scaled_dot_product_attention` for efficient computation.

!!! warning "Channel constraint"
    `inChannels` must be divisible by `numHeads`. A `ValueError` is raised otherwise.

---

### DownSampleBlock

:material-file-code: `blocks/downsample.py`

Strided convolution for spatial downsampling.

```python
class DownSampleBlock(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, stride=2, padding=1)
```

**Input:** `[B, C_in, H, W]` → **Output:** `[B, C_out, H/2, W/2]` (with defaults)

---

### UpSampleBlock

:material-file-code: `blocks/upsample.py`

Nearest-neighbor interpolation (2×) followed by convolution.

```python
class UpSampleBlock(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, padding=1)
```

**Input:** `[B, C_in, H, W]` → **Output:** `[B, C_out, 2H, 2W]` (with defaults)

---

## Type Definitions

:material-file-code: `models/networks/blocks/__init__.py`

Type aliases used in U-Net configuration:

```python
EncoderItem = Union[
    Tuple[Literal["ResNet"], ResBlockParams],
    Tuple[Literal["DownSample"], UpDownBlockParams],
    Tuple[Literal["Attention"], AttentionBlockParams]
]

BottleneckItem = Union[
    Tuple[Literal["ResNet"], ResBlockParams],
    Tuple[Literal["Attention"], AttentionBlockParams]
]

DecoderItem = Union[
    Tuple[Literal["ResNet"], ResBlockParams],
    Tuple[Literal["UpSample"], UpDownBlockParams],
    Tuple[Literal["Attention"], AttentionBlockParams]
]
```
