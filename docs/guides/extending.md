# Extending the Framework

This guide shows how to add new components to the framework. The registry-driven architecture makes it straightforward to add new models, datasets, noise schedules, and training components without modifying any existing infrastructure code.

---

## Adding a New Diffusion Model

To add a new diffusion model (e.g., DDIM), follow these steps:

### Step 1: Create the Model Package

Create a new directory under `models/`:

```
models/
├── ddim/                   # ← New!
│   ├── __init__.py
│   ├── ddim.py
│   └── config.py
```

### Step 2: Implement the Model Class

Subclass `BaseDiffusion` and implement the `forward()` and `sample()` methods:

```python title="models/ddim/ddim.py"
import torch
from typing import Tuple
from models.base.diffusion import BaseDiffusion
from models.networks.unet import UNet
from models.base.noiseSchedule import NoiseSchedule


class DDIM(BaseDiffusion):

    def __init__(self, unetModel: UNet, noiseSchedule: NoiseSchedule, T: int = 1000):
        super().__init__(noiseSchedule=noiseSchedule, T=T)
        self.unet = unetModel

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Training forward pass — could be same as DDPM
        t = torch.randint(0, self.T, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        noisyImages = self.signalCoefficients[t].view(-1,1,1,1) * x \
                    + self.noiseCoefficients[t].view(-1,1,1,1) * noise
        return (noise, self.unet(noisyImages, t))

    @torch.no_grad()
    def sample(self, batchSize, imageSize, device):
        # DDIM-specific deterministic sampling
        ...
```

!!! info "What `BaseDiffusion` gives you"
    The base class automatically precomputes and registers all diffusion coefficients (`alphas`, `cumulativeAlphas`, `signalCoefficients`, `noiseCoefficients`, `posteriorVariance`, etc.) as buffers. You just use them in your `forward()` and `sample()` methods.

### Step 3: Define the Architecture Config

```python title="models/ddim/config.py"
from models.networks.blocks import EncoderItem, DecoderItem, BottleneckItem
from typing import List

initChannels = 128

encoderConfig: List[EncoderItem] = [
    # Define your architecture here
    ("ResNet", {"outChannels": 128}),
    ...
]

bottleneckConfig: List[BottleneckItem] = [...]
decoderConfig: List[DecoderItem] = [...]
```

### Step 4: Register the Model

Create the builder function and register:

```python title="models/ddim/__init__.py"
from models import registerModel
from models.ddim.ddim import DDIM
from models.ddim.config import encoderConfig, bottleneckConfig, decoderConfig, initChannels
from models.networks.embeddings import TimeStepEmbedding
from models.networks.unet import UNet
from tools.trainingUtils import selectNoiseScheduler


def buildDDIM(config):
    noiseSchedule = selectNoiseScheduler(
        config.model.varianceSchedule, 
        **config.model.varianceScheduleParams
    )
    timeEmbedding = TimeStepEmbedding(config.model.timeEmbeddingDimension)
    unetModel = UNet(
        timeStepEmbedding=timeEmbedding,
        initChannels=initChannels,
        encoder=encoderConfig,
        bottleneck=bottleneckConfig,
        decoder=decoderConfig
    )
    return DDIM(unetModel=unetModel, noiseSchedule=noiseSchedule, T=config.model.timesteps)


registerModel("ddim", builder=buildDDIM)(DDIM)
```

### Step 5: Trigger Registration

Import the new model package in `models/__init__.py` or — if using lazy imports — ensure the import happens. The current approach is to import in the model's own `__init__.py` and have the config reference `name = "ddim"`.

### Step 6: Create a Config File

```python title="configs/ddim_celebahq.py"
from configs.schema import Config, ModelConfig, DatasetConfig, TrainingConfig

config = Config(
    model = ModelConfig(name="ddim", ...),
    dataset = DatasetConfig(...),
    training = TrainingConfig(...)
)
```

### Step 7: Train

```bash
python train.py --config configs.ddim_celebahq
```

**Zero changes** to `train.py` or `inference.py` required! :tada:

---

## Adding a New Dataset

### Step 1: Create the Dataset Module

```python title="datasets/cifar10.py"
from datasets import registerDataset
from torch.utils.data import Dataset
import torchvision


@registerDataset('cifar10')
class CIFAR10Wrapper(Dataset):

    def __init__(self, rootPath, download=False, transforms=None):
        self.dataset = torchvision.datasets.CIFAR10(
            root=rootPath, train=True, download=download, transform=transforms
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Ignore labels for unconditional generation
        return image
```

### Step 2: Register It

Add the import in `datasets/__init__.py`:

```python title="datasets/__init__.py"
# At the bottom of the file
import datasets.cifar10    # ← Add this line
```

### Step 3: Use It in a Config

```python
dataset = DatasetConfig(
    name = "cifar10",
    rootPath = "./datasets",
    download = True,
)
```

---

## Adding a New Noise Schedule

### Step 1: Implement the Schedule

```python title="models/base/noiseSchedule.py"
class CosineSchedule(NoiseSchedule):
    """
    Cosine variance schedule as proposed in Improved DDPM (Nichol & Dhariwal, 2021).
    """

    def __init__(self, s: float = 0.008):
        self.s = s

    def __call__(self, T: int) -> torch.Tensor:
        steps = torch.arange(T + 1, dtype=torch.float32)
        f = torch.cos(((steps / T) + self.s) / (1 + self.s) * (torch.pi / 2)) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
```

### Step 2: Register It

Add a case in the `selectNoiseScheduler` function in `tools/trainingUtils.py`:

```python
def selectNoiseScheduler(scheduleType: str, **kwargs):
    if scheduleType == "linear":
        return LinearSchedule(**kwargs)
    elif scheduleType == "cosine":
        return CosineSchedule(**kwargs)
    else:
        raise NotImplementedError(f"Schedule '{scheduleType}' not yet implemented.")
```

### Step 3: Use It

```python
model = ModelConfig(
    varianceSchedule = "cosine",
    varianceScheduleParams = {"s": 0.008},
)
```

---

## Adding a New Optimizer / Loss / Scheduler

These are the simplest extensions — just add an entry to the appropriate registry in `tools/trainingUtils.py`:

### Optimizer

```python
from torch.optim import RAdam

OPTIMIZER_REGISTRY = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "radam": RAdam,          # ← Add this
}
```

### Loss Function

```python
LOSS_REGISTRY = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "huber": nn.HuberLoss,
    "smooth_l1": nn.SmoothL1Loss,
    "charbonnier": CharbonnierLoss,  # ← Add your custom loss
}
```

### LR Scheduler

```python
from torch.optim.lr_scheduler import OneCycleLR

SCHEDULER_REGISTRY = {
    "cosine": CosineAnnealingLR,
    "step": StepLR,
    "exponential": ExponentialLR,
    "linear": LinearLR,
    "onecycle": OneCycleLR,  # ← Add this
}
```

Then use the new component name in your config:

```python
training = TrainingConfig(
    optimizer = "radam",
    scheduler = "onecycle",
    schedulerParams = {"max_lr": 0.001, "total_steps": 10000},
)
```
