# ====================================== LIBRARIES ======================================
from models.base.noiseSchedule import *

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD, RMSprop, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, LinearLR, LRScheduler

import platform
from typing import List, Optional
# =======================================================================================


# ====================================== FACTORY MAPS ===================================

OPTIMIZER_REGISTRY = {
                            "adam": Adam,
                            "adamw": AdamW,
                            "sgd": SGD,
                            "rmsprop": RMSprop,
                     }


LOSS_REGISTRY = {
                            "mse": nn.MSELoss,
                            "l1": nn.L1Loss,
                            "huber": nn.HuberLoss,
                            "smooth_l1": nn.SmoothL1Loss,
                }


SCHEDULER_REGISTRY = {
                            "cosine": CosineAnnealingLR,
                            "step": StepLR,
                            "exponential": ExponentialLR,
                            "linear": LinearLR,
                     }


AMP_DTYPE_MAP = {
                        "fp16": torch.float16,
                        "float16": torch.float16,
                        "half": torch.float16,

                        "bf16": torch.bfloat16,
                        "bfloat16": torch.bfloat16,

                        "none": None

                    }

# =======================================================================================


def selectAMPType(ampType: str) -> torch.dtype | None:

    '''
        Convert the specific torch dtype str in the corresponding object for the training.

         Parameters
        ----------
         ampType: str
          The Case-insensitive dtype format to use for the Automatic Mixed Precision (i.e. float16 or bfloat16)
    '''

    if ampType:
        return AMP_DTYPE_MAP[ampType.lower()]
    
    else:
        return None
    

def selectNoiseScheduler(scheduleType: str, **kwargs):

    '''
        Define the Noise Scheduler object according to the provided parameters present in the configuration file.

        Parameters
        ----------
         scheduleType: str
          The Noise Scheduler Type to use (e.g. 'linear')

         **kwargs
          The extra keywords argument containing all the parameters required by the specific noise scheduler used

    '''

    if scheduleType == "linear":
        noiseSchedule = LinearSchedule(**kwargs)

    else:
        raise NotImplementedError(f"Schedule '{scheduleType}' not yet implemented.")
    
    return noiseSchedule


def buildOptimizer(name: str, params, lr: float, **kwargs) -> Optimizer:

    '''
        Build an optimizer from its name and parameters.

        Parameters
        ----------
         name : str
          Case-insensitive optimizer name (e.g. 'AdamW', 'sgd')

         params
          Model parameters (from model.parameters())

         lr : float
          Learning rate

         **kwargs
          Any extra keyword arguments forwarded to the optimizer constructor (e.g. weight_decay, momentum)- See the specific Optimizer on the PyTorch documentation for these parameters)

        Returns
        -------
        torch.optim.Optimizer
    '''

    key = name.lower()

    if key not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer '{name}' is not supported. The available optimizers are: {list(OPTIMIZER_REGISTRY.keys())}")

    return OPTIMIZER_REGISTRY[key](params, lr = lr, **kwargs)


def buildLossFunction(name: str) -> nn.Module:

    '''
        Build a loss function from its name.

        Parameters
        ----------
         name : str
          Case-insensitive loss name (e.g. 'MSE', 'l1', 'huber').

        Returns
        -------
        torch.nn.Module
    '''

    key = name.lower()

    if key not in LOSS_REGISTRY:
        raise ValueError(f"Loss '{name}' is not supported. The available losses are: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[key]()


def buildScheduler(name: str, optimizer: Optimizer, **kwargs) -> Optional[LRScheduler]:

    '''
        Build a learning rate scheduler from its name.

        Parameters
        ----------
         name : str
          Case-insensitive scheduler name (e.g. 'cosine', 'step').
          Pass an empty string or None to skip scheduler creation.

         optimizer : torch.optim.Optimizer
          The optimizer whose learning rate will be scheduled.

         **kwargs
          Extra keyword arguments forwarded to the scheduler constructor
          (e.g. T_max, step_size, gamma - See the specific Optimizer on the PyTorch documentation for these parameters).

        Returns
        -------
        torch.optim.lr_scheduler.LRScheduler or None
    '''

    if not name:
        return None

    key = name.lower()

    if key not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{name}' is not supported. Available schedulers: {list(SCHEDULER_REGISTRY.keys())}")
    
    return SCHEDULER_REGISTRY[key](optimizer, **kwargs)


def getDevice() -> torch.device:

    '''
        Automatically selects the best available device for PyTorch.

        Priority
        --------
            1. CUDA (NVIDIA GPU)
            2. MPS (Apple Silicon)
            3. CPU (fallback)

        Returns
        -------
        torch.device
         The selected device.
    '''

    if torch.cuda.is_available():
        return torch.device("cuda")
    
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")