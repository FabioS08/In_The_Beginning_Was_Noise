# ====================================== LIBRARIES ======================================
import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Literal, Tuple
from models.UNet import UNet
# =======================================================================================


class DDPM(nn.Module):

    '''
        The core DDPM framework that handles the Forward and Reverse Diffusion Processes.

        Parameters
        ----------
        model: nn.Module
         The U-Net model used to predict the noise.

        varianceSchedule: Tuple
         A tuple containing the type of variance schedule and its parameters.
         Up to now, you can use:

                - ('linear', betaStart, betaEnd): A linear variance schedule that starts from betaStart and ends at betaEnd.
         
         [Default: ('linear', 1e-4, 0.02)]
        
        timesteps: int
         The total number of diffusion steps (T). [Default: 1000]

        Methods
        -------
        forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
         The forward pass of the model that computes the noise added to the images at a random timestep and the noise prediction of the U-Net, both element will be returned as output.

        sample(batchSize: int, imageSize: Tuple[int, int] | int, device: torch.device) -> torch.Tensor:
         The inference process that genereates new samples by iteratively denoising pure noise samples from T to 0.
         It will take in input the batch size, the image size (either int for square images or a tuple of two int for rectangular ones) and the device where to perform the inference, and it will return the generated samples as output.

    '''

    def __init__(self, unetModel: UNet, varianceSchedule: Tuple[str, float, float] = ('linear', 1e-4, 0.02), T: int = 1000,):

        super().__init__()

        self.unet = unetModel
        self.T = T

        self.defineVarianceSchedule(varianceSchedule)
        self.computeCoefficients()


    def forward(self, x: torch.Tensor):
        
        # Sample a random timestep for the noise addition of each image in the batch
        t = torch.randint(0, self.T, (x.shape[0], ), device = x.device)
        noise = torch.randn(x.shape, device = x.device)

        # Compute the noisy images at time t through the closed-form approach
        noisyImages = self.signalCoefficients[t].view(-1, 1, 1, 1) * x + self.noiseCoefficients[t].view(-1, 1, 1, 1) * noise

        return(noise, self.unet(noisyImages, t))
    

    @torch.no_grad()
    def sample(self, batchSize: int, imageSize: Tuple[int, int] | int, device: torch.device):

        # Generate the Pure Noise Samples
        if isinstance(imageSize, int):
            x = torch.randn((batchSize, 3, imageSize, imageSize), device = device)

        else:
            x = torch.randn((batchSize, 3, imageSize[0], imageSize[1]), device = device)

        # Iteratively denoising the samples from T to 0
        for t in reversed(range(self.T)):

            # Use the U-Net to infer the noise component for each sample at time t
            noiseComponent = self.unet(x, torch.full((batchSize, ), t, device = device))

            # Compute the mean
            mean = self.reciprocalSQRTAlphas[t].view(-1, 1, 1, 1) * (x - self.reverseNoiseCoefficient[t].view(-1, 1, 1, 1) * noiseComponent)

            # Sample the noise to add if not last timestep
            if t > 0:
                noise = torch.randn(x.shape, device = device)
            
            else:
                noise = torch.zeros(x.shape, device = device)

            # Compute the sample at the timestep t - 1
            x = mean + torch.sqrt(self.posteriorVariance[t]) * noise

        return(x)
    

    def defineVarianceSchedule(self, varianceSchedule):

        # varianceSchedule = ('linear', betaStart, betaEnd)
        # ...To be extended

        scheduleType = varianceSchedule[0]
        
        if scheduleType == 'linear':
            
            betaStart, betaEnd = varianceSchedule[1], varianceSchedule[2]
            self.betas: torch.Tensor
            self.register_buffer('betas', torch.linspace(betaStart, betaEnd, self.T))

        else: 

            raise NotImplementedError(f"The Schedule '{scheduleType}' has not been implemented yet!")
        

    def computeCoefficients(self):

        # Parameters used during the Forward Diffusion Process
        self.alphas: torch.Tensor
        self.cumulativeAlphas: torch.Tensor
        self.signalCoefficients: torch.Tensor
        self.noiseCoefficients: torch.Tensor

        # Parameters used during the Reverse Diffusion Process
        self.reciprocalSQRTAlphas: torch.Tensor
        self.reverseNoiseCoefficient: torch.Tensor
        self.posteriorVariance: torch.Tensor

        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('cumulativeAlphas', torch.cumprod(self.alphas, dim = 0))
        self.register_buffer('signalCoefficients', torch.sqrt(self.cumulativeAlphas))
        self.register_buffer('noiseCoefficients', torch.sqrt(1 - self.cumulativeAlphas))
        
        self.register_buffer('reciprocalSQRTAlphas', torch.sqrt(1 / self.alphas))
        self.register_buffer('reverseNoiseCoefficient', (1 - self.alphas) * torch.sqrt(1 / (1 - self.cumulativeAlphas)))

        cumulativeAlphasPrev = F.pad(self.cumulativeAlphas[:-1], (1, 0), value=1.0)
        self.register_buffer('posteriorVariance', 
        self.betas * (1 - cumulativeAlphasPrev) / (1 - self.cumulativeAlphas))