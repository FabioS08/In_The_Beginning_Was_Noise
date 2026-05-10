# Tools API

:material-file-code: `tools/`

---

## DiffusionTrainer

:material-file-code: `tools/trainer.py`

The main training orchestrator. Manages the training loop, logging, checkpointing, gradient accumulation, and mixed precision.

```python
class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        lossFunction: nn.Module,
        device: torch.device,
        ampType: Optional[torch.dtype],
        gradientAccumulationSteps: int = 1,
        checkpointSavingFrequency: int = 5,
        maxNumCheckpoints: int = 1,
        checkpointPathRestart: Optional[Path] = None,
        scheduler: Optional[LRScheduler] = None,
        maxGradNorm: Optional[float] = 10.0,
        logPath: Optional[Path] = None,
        checkpointDirectory: Optional[Path] = None
    )
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model` | `nn.Module` | — | The diffusion model |
| `dataloader` | `DataLoader` | — | Training data loader |
| `optimizer` | `Optimizer` | — | Optimizer instance |
| `lossFunction` | `nn.Module` | — | Loss function (e.g. `nn.MSELoss()`) |
| `device` | `torch.device` | — | Training device |
| `ampType` | `torch.dtype` or `None` | — | AMP dtype (`float16`, `bfloat16`, or `None`) |
| `gradientAccumulationSteps` | `int` | `1` | Accumulate gradients over N batches |
| `checkpointSavingFrequency` | `int` | `5` | Save every N epochs |
| `maxNumCheckpoints` | `int` | `1` | Max checkpoints to keep on disk |
| `checkpointPathRestart` | `Path` or `None` | `None` | Resume from this checkpoint |
| `scheduler` | `LRScheduler` or `None` | `None` | Learning rate scheduler |
| `maxGradNorm` | `float` or `None` | `10.0` | Gradient clipping threshold |
| `logPath` | `Path` or `None` | `None` | Custom log file path |
| `checkpointDirectory` | `Path` or `None` | `None` | Custom checkpoint directory |

### `train(epochs: int)`

Runs the full training loop from `startEpoch` to `epochs`.

**Features:**

- Dual logging (Rich console + plain-text file)
- Optional mixed precision via `torch.autocast` + `GradScaler`
- Gradient accumulation with proper scaling
- Gradient norm clipping
- Checkpoint rotation (keeps only the N most recent)
- Auto-resume from checkpoint
- ETA tracking (per-epoch and global)

### Private Methods

| Method | Description |
|:-------|:------------|
| `_defineSavingPaths()` | Creates `logs/{Model}/{timestamp}/` directory structure |
| `_attachFileLogger()` | Sets up the plain-text file logger alongside Rich |
| `_saveCheckpoint()` | Saves model + optimizer + scaler + scheduler state |
| `_resumeFromCheckpoint()` | Loads and restores all training state |
| `_printTrainConfiguration()` | Prints model architecture and training params in Rich panels |

---

## CLI Classes

:material-file-code: `tools/cli.py`

### BaseCLI (ABC)

Abstract base class providing structured argument parsing and Rich-formatted help output.

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `title` | `str` | Title shown in the help panel |
| `description` | `str` | Description text |
| `note` | `str` | Optional note (shown dimmed) |
| `usageExample` | `str` | Example command shown in help |

#### Methods

| Method | Description |
|:-------|:------------|
| `_setupArguments()` | (Abstract) Define argparse arguments |
| `_getHelpRows()` | (Abstract) Return `(arg, description)` rows for the help table |
| `_printHelp()` | Render the Rich-formatted help panel |
| `_loadConfig(moduleName)` | Dynamically import a config module |
| `parse()` | Parse arguments and return the result |

---

### TrainCLI

CLI handler for `train.py`.

| Argument | Type | Description |
|:---------|:-----|:------------|
| `--config` | `str` | Config module path |
| `--printConfig` | flag | Print config and exit |
| `-h, --help` | flag | Show help |

**`parse() → Config`**: Returns the loaded `Config` dataclass.

---

### InferenceCLI

CLI handler for `inference.py`.

| Argument | Type | Default | Description |
|:---------|:-----|:--------|:------------|
| `--config` | `str` | — | Config module path |
| `--checkpoint` | `str` | — | Checkpoint file path |
| `--printConfig` | flag | — | Print config and exit |
| `--outputDir` | `str` | `"."` | Output directory for images |
| `--numImages` | `int` | `1` | Number of images to generate |
| `--imageSize` | `int` or `tuple` | `256` | Image size (e.g. `256` or `256,512`) |
| `-h, --help` | flag | — | Show help |

**`parse() → Namespace`**: Returns argparse namespace with `config` field as a `Config` dataclass.

---

## Training Utilities

:material-file-code: `tools/trainingUtils.py`

### `getDevice() → torch.device`

Auto-selects the best available device.

**Priority:** CUDA → MPS (Apple Silicon) → CPU

---

### `selectNoiseScheduler(scheduleType, **kwargs) → NoiseSchedule`

Build a noise schedule from its name.

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `scheduleType` | `str` | Schedule name (`"linear"`) |
| `**kwargs` | | Forwarded to schedule constructor |

---

### `buildOptimizer(name, params, lr, **kwargs) → Optimizer`

Build an optimizer from its name.

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `name` | `str` | Case-insensitive optimizer name |
| `params` | | Model parameters |
| `lr` | `float` | Learning rate |
| `**kwargs` | | Extra args (e.g. `weight_decay`) |

---

### `buildLossFunction(name) → nn.Module`

Build a loss function from its name.

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `name` | `str` | Case-insensitive loss name |

---

### `buildScheduler(name, optimizer, **kwargs) → LRScheduler | None`

Build a learning rate scheduler. Returns `None` if `name` is empty.

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `name` | `str` | Case-insensitive scheduler name |
| `optimizer` | `Optimizer` | The optimizer to schedule |
| `**kwargs` | | Extra args (e.g. `T_max`, `step_size`) |

---

### `selectAMPType(ampType) → torch.dtype | None`

Convert an AMP type string to the corresponding `torch.dtype`.

| Input | Output |
|:------|:-------|
| `"fp16"`, `"float16"`, `"half"` | `torch.float16` |
| `"bf16"`, `"bfloat16"` | `torch.bfloat16` |
| `"none"`, `""` | `None` |
