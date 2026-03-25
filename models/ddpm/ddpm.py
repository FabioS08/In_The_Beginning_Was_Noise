# ====================================== LIBRARIES ======================================
import torch
import torch.nn as nn
from typing import Tuple

from models.base.diffusion import BaseDiffusion
from models.base.noiseSchedule import NoiseSchedule
from models.networks.unet import UNet
# =======================================================================================


class DDPM(BaseDiffusion):

    '''
        The core DDPM framework that handles the Forward and Reverse Diffusion Processes.

        Parameters
        ----------
        unetModel : UNet
         The U-Net model used to predict the noise.

        noiseSchedule : NoiseSchedule
         The noise schedule defining how betas evolve over timesteps.

        T : int
         The total number of diffusion steps (T). [Default: 1000]

        Methods
        -------
        `forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`:
         The forward pass of the model that computes the noise added to the images at a random timestep and the noise prediction of the U-Net, both element will be returned as output.

        `sample(batchSize: int, imageSize: Tuple[int, int] | int, device: torch.device) -> torch.Tensor`:
         The inference process that genereates new samples by iteratively denoising pure noise samples from T to 0.
         It will take in input the batch size, the image size (either int for square images or a tuple of two int for rectangular ones) and the device where to perform the inference, and it will return the generated samples as output.

    '''

    def __init__(self, unetModel: UNet, noiseSchedule: NoiseSchedule, T: int = 1000):

        super().__init__(noiseSchedule = noiseSchedule, T = T)

        self.unet = unetModel


    def forward(self, x: torch.Tensor):
        
        # Sample a random timestep for the noise addition of each image in the batch
        t = torch.randint(0, self.T, (x.shape[0], ), device = x.device)
        noise = torch.randn(x.shape, device = x.device)

        # Compute the noisy images at time t through the closed-form approach
        noisyImages = self.signalCoefficients[t].view(-1, 1, 1, 1) * x + self.noiseCoefficients[t].view(-1, 1, 1, 1) * noise

        return (noise, self.unet(noisyImages, t))
    

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
