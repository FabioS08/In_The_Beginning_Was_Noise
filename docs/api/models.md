# Models API

:material-file-code: `models/`

---

## Model Registry

:material-file-code: `models/__init__.py`

The central mechanism for registering and retrieving model classes. All models self-register when their package is imported.

### `registerModel(name, builder=None)`

Decorator that registers a model class in the global `MODEL_REGISTRY`.

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `name` | `str` | Registry key (e.g. `"ddpm"`) |
| `builder` | `Callable`, optional | Factory function `builder(config) → nn.Module` |

```python
@registerModel("ddpm", builder=buildDDPM)
class DDPM(BaseDiffusion):
    ...
```

---

### `getModel(name) → type`

Retrieve a registered model **class** by name.

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `name` | `str` | Registered model name |

**Raises:** `KeyError` if the name is not found.

---

### `buildModel(config) → nn.Module`

Build a fully constructed model from a configuration object. Looks up the builder by `config.model.name`.

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `config` | `Config` | Top-level configuration dataclass |

**Raises:** `KeyError` if model not found, `ValueError` if no builder registered.

---

## BaseDiffusion

:material-file-code: `models/base/diffusion.py`

Abstract base class for all diffusion models. Handles noise schedule management and coefficient registration.

```python
class BaseDiffusion(nn.Module):
    def __init__(self, noiseSchedule: NoiseSchedule, T: int = 1000)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `noiseSchedule` | `NoiseSchedule` | — | The noise schedule object |
| `T` | `int` | `1000` | Total diffusion timesteps |

### Registered Buffers

All buffers are 1D tensors of shape `[T]`:

| Buffer | Formula | Description |
|:-------|:--------|:------------|
| `betas` | \( \beta_t \) | Noise variances from schedule |
| `alphas` | \( \alpha_t = 1 - \beta_t \) | Signal retention per step |
| `cumulativeAlphas` | \( \bar{\alpha}_t = \prod_{s=1}^t \alpha_s \) | Cumulative signal retention |
| `signalCoefficients` | \( \sqrt{\bar{\alpha}_t} \) | Scales clean image in forward |
| `noiseCoefficients` | \( \sqrt{1 - \bar{\alpha}_t} \) | Scales noise in forward |
| `reciprocalSQRTAlphas` | \( 1 / \sqrt{\alpha_t} \) | Used in reverse mean |
| `reverseNoiseCoefficient` | \( (1-\alpha_t) / \sqrt{1-\bar{\alpha}_t} \) | Used in reverse mean |
| `posteriorVariance` | \( \tilde{\beta}_t \) | Posterior variance for sampling |

### Abstract Methods

#### `forward(x) → Tuple[Tensor, Tensor]`

Training forward pass. Must return `(target, prediction)`.

#### `sample(batchSize, imageSize, device) → Tensor`

Generate samples by running the reverse process.

---

## DDPM

:material-file-code: `models/ddpm/ddpm.py`

Implementation of *Denoising Diffusion Probabilistic Models* (Ho et al., 2020).

```python
class DDPM(BaseDiffusion):
    def __init__(self, unetModel: UNet, noiseSchedule: NoiseSchedule, T: int = 1000)
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `unetModel` | `UNet` | The noise prediction network |
| `noiseSchedule` | `NoiseSchedule` | Variance schedule |
| `T` | `int` | Number of timesteps |

### `forward(x) → Tuple[Tensor, Tensor]`

1. Samples random timestep \( t \) per image
2. Computes \( \mathbf{x}_t \) via closed-form
3. Returns `(noise, predicted_noise)`

| Input | Shape | Description |
|:------|:------|:------------|
| `x` | `[B, 3, H, W]` | Batch of clean images |

| Output | Shape | Description |
|:-------|:------|:------------|
| `noise` | `[B, 3, H, W]` | Ground truth noise |
| `prediction` | `[B, 3, H, W]` | U-Net predicted noise |

### `sample(batchSize, imageSize, device) → Tensor`

Runs the reverse diffusion process (Algorithm 2 from the paper).

| Input | Type | Description |
|:------|:-----|:------------|
| `batchSize` | `int` | Number of images to generate |
| `imageSize` | `int` or `Tuple[int, int]` | Spatial dimensions |
| `device` | `torch.device` | Device for computation |

**Returns:** Tensor of shape `[B, 3, H, W]` with generated images (values in `[-1, 1]`).

---

## NoiseSchedule

:material-file-code: `models/base/noiseSchedule.py`

### `NoiseSchedule` (ABC)

Abstract base class for noise schedules.

#### `__call__(T: int) → Tensor`

Returns a 1D tensor of shape `[T]` containing the beta values.

---

### `LinearSchedule`

Linear variance schedule from \( \beta_\text{start} \) to \( \beta_\text{end} \).

```python
class LinearSchedule(NoiseSchedule):
    def __init__(self, betaStart: float = 1e-4, betaEnd: float = 0.02)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `betaStart` | `float` | `1e-4` | Starting beta value |
| `betaEnd` | `float` | `0.02` | Ending beta value |
