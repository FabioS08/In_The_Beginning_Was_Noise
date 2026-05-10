# Configuration Reference

All training and inference behavior is controlled through **Python configuration files** using typed dataclasses. Configuration files live in the `configs/` directory.

---

## Config Schema

The top-level `Config` dataclass contains three sections:

```python
@dataclass
class Config:
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
```

---

## ModelConfig

Controls the diffusion model parameters.

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `name` | `str` | `"ddpm"` | Model name (must match a registered model in the registry) |
| `timesteps` | `int` | `1000` | Total number of diffusion timesteps \( T \) |
| `varianceSchedule` | `str` | `"linear"` | Noise schedule type |
| `varianceScheduleParams` | `dict` | `{"betaStart": 1e-4, "betaEnd": 0.02}` | Parameters for the noise schedule |
| `timeEmbeddingDimension` | `int` | `128` | Base dimension for sinusoidal timestep embedding (must be even). Output dim = 4× this value. |

!!! info "Variance Schedule Options"
    Currently only `"linear"` is implemented. The schedule params are passed as `**kwargs` to the schedule constructor.

---

## DatasetConfig

Controls dataset loading.

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `name` | `str` | `"celebahq"` | Dataset name (must match a registered dataset) |
| `rootPath` | `str` | `"./datasets"` | Root directory for dataset storage |
| `download` | `bool` | `False` | Whether to download the dataset if not present |

!!! warning "Download"
    For CelebA-HQ, the dataset is downloaded from Google Drive (~1.3 GB). Make sure you have enough disk space and a stable connection. After the first download, set `download=False`.

---

## TrainingConfig

Controls all training hyperparameters.

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `batchSize` | `int` | `16` | Batch size for training |
| `epochs` | `int` | `100` | Total number of training epochs |
| `learningRate` | `float` | `0.0001` | Initial learning rate |
| `optimizer` | `str` | `"adamw"` | Optimizer name (case-insensitive) |
| `optimizerParams` | `dict` | `{}` | Extra kwargs passed to the optimizer (e.g. `{"weight_decay": 0.01}`) |
| `loss` | `str` | `"mse"` | Loss function name |
| `scheduler` | `str` | `""` | LR scheduler name (empty = no scheduler) |
| `schedulerParams` | `dict` | `{}` | Extra kwargs passed to the scheduler (e.g. `{"T_max": 100}`) |
| `gradientAccumulationSteps` | `int` | `1` | Number of gradient accumulation steps |
| `ampType` | `str` | `""` | Mixed precision type (`"fp16"`, `"bf16"`, or `""` for disabled) |
| `checkpointSavingFrequency` | `int` | `5` | Save a checkpoint every N epochs |
| `maxNumCheckpoints` | `int` | `1` | Maximum checkpoints to keep (oldest deleted first) |
| `checkpointPathRestart` | `str` | `""` | Path to checkpoint for resuming training |
| `maxGradNorm` | `float` | `10.0` | Maximum gradient norm for clipping |
| `checkpointDirectory` | `str` | `""` | Custom directory for saving checkpoints |

---

## Example: Full Configuration File

```python title="configs/ddpm_celebahq.py"
from configs.schema import Config, ModelConfig, DatasetConfig, TrainingConfig

config = Config(

    model = ModelConfig(
        name = "ddpm",
        timesteps = 1000,
        varianceSchedule = "linear",
        varianceScheduleParams = {"betaStart": 1e-4, "betaEnd": 0.02},
        timeEmbeddingDimension = 128,
    ),

    dataset = DatasetConfig(
        name = "celebahq",
        rootPath = "./datasets",
        download = False,
    ),

    training = TrainingConfig(
        batchSize = 1,
        epochs = 500,
        learningRate = 0.0001,
        optimizer = 'AdamW',
        optimizerParams = {},
        scheduler = '',
        schedulerParams = {},
        loss = 'MSE',
        gradientAccumulationSteps = 1,
        ampType = 'float16',
        checkpointSavingFrequency = 5,
        maxNumCheckpoints = 2,
        checkpointPathRestart = '',
        maxGradNorm = 10,
        checkpointDirectory = ''
    )
)
```

---

## U-Net Architecture Configuration

Besides the training config, each model has its own **architecture configuration** in `models/{model}/config.py`. For DDPM, this defines the U-Net structure:

```python title="models/ddpm/config.py"
initChannels = 128  # Entry convolution output channels

encoderConfig = [
    ("ResNet",     {"outChannels": 128}),
    ("ResNet",     {"outChannels": 128}),
    ("DownSample", {"outChannels": 128}),
    # ...
]

bottleneckConfig = [
    ("ResNet",    {"outChannels": 512}),
    ("Attention", {"numHeads": 1}),
    ("ResNet",    {"outChannels": 512}),
]

decoderConfig = [
    ("ResNet",   {"outChannels": 512}),
    ("ResNet",   {"outChannels": 512}),
    ("ResNet",   {"outChannels": 512}),
    ("UpSample", {"outChannels": 512}),
    # ...
]
```

!!! tip "For the full specification"
    See the [U-Net Architecture](../architecture/unet.md) page for details on block types, parameter options, and the skip connection alignment rules.

---

## Creating a New Configuration

To create a new configuration (e.g. for a different dataset or different hyperparameters):

1. Create a new file in `configs/`, e.g. `configs/ddpm_cifar10.py`
2. Import the schema:
   ```python
   from configs.schema import Config, ModelConfig, DatasetConfig, TrainingConfig
   ```
3. Define a `config` variable with your settings
4. Use it:
   ```bash
   python train.py --config configs.ddpm_cifar10
   ```

!!! warning "The `config` variable"
    The configuration file **must** contain a variable called `config`. This is loaded dynamically by the CLI parser.
