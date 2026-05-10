# ====================================== LIBRARIES ======================================
import torch
import torch.nn as nn
from torch.nn import functional as F

from abc import abstractmethod
from typing import Tuple

from models.base.noiseSchedule import NoiseSchedule
# =======================================================================================


class BaseDiffusion(nn.Module):

    '''
        Abstract base class for diffusion models.

        Provides shared infrastructure for the diffusion process, including:

        - Noise schedule management
        - Pre-computation and registration of diffusion coefficients (alphas, cumulative products, etc...)
        
        Subclasses must implement `forward()` and `sample()` with their model-specific forward/reverse diffusion logic.

        Parameters
        ----------
        noiseSchedule: NoiseSchedule
            The noise schedule object that defines how betas evolve over timesteps.

        T: int
            The total number of diffusion timesteps. [Default: 1000]
    '''

    def __init__(self, noiseSchedule: NoiseSchedule, T: int = 1000):

        super().__init__()

        self.T = T
        self.noiseSchedule = noiseSchedule

        # Compute and register the betas and all derived coefficients
        betas = noiseSchedule(T)
        self._registerCoefficients(betas)


    def _registerCoefficients(self, betas: torch.Tensor):

        '''
            Pre-compute and register all diffusion coefficients as buffers.

            These are shared across all diffusion models and include:

            - Forward process coefficients (alphas, cumulative products, signal/noise scaling)
            - Reverse process coefficients (posterior variance, denoising coefficients)

            Parameters
            ----------
            betas : torch.Tensor
                A 1D tensor of shape [T] containing the beta values from the noise schedule.
        '''

        # Forward Process Coefficients
        self.betas: torch.Tensor
        self.alphas: torch.Tensor
        self.cumulativeAlphas: torch.Tensor
        self.signalCoefficients: torch.Tensor
        self.noiseCoefficients: torch.Tensor

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('cumulativeAlphas', torch.cumprod(self.alphas, dim = 0))
        self.register_buffer('signalCoefficients', torch.sqrt(self.cumulativeAlphas))
        self.register_buffer('noiseCoefficients', torch.sqrt(1 - self.cumulativeAlphas))
        
        # Reverse Process Coefficients
        self.reciprocalSQRTAlphas: torch.Tensor
        self.reverseNoiseCoefficient: torch.Tensor
        self.posteriorVariance: torch.Tensor

        self.register_buffer('reciprocalSQRTAlphas', torch.sqrt(1 / self.alphas))
        self.register_buffer('reverseNoiseCoefficient', (1 - self.alphas) * torch.sqrt(1 / (1 - self.cumulativeAlphas)))

        cumulativeAlphasPrev = F.pad(self.cumulativeAlphas[:-1], (1, 0), value=1.0)
        self.register_buffer('posteriorVariance', 
            self.betas * (1 - cumulativeAlphasPrev) / (1 - self.cumulativeAlphas))


    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        '''
            The training forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input images of shape [BatchSize, C, H, W].

            Returns
            -------
            Model-specific outputs (e.g. target noise and predicted noise).

        '''
        ...


    @abstractmethod
    def sample(self, batchSize: int, imageSize: Tuple[int, int] | int, device: torch.device) -> torch.Tensor:

        '''
            Generate new samples by iteratively denoising from pure noise.

            Parameters
            ----------
            batchSize : int
                Number of samples to generate.

            imageSize : Tuple[int, int] | int
                Spatial dimensions of the generated images.

            device : torch.device
                Device to perform inference on.

            Returns
            -------
            torch.Tensor
                Generated samples of shape [BatchSize, C, H, W].
        '''
        ...
