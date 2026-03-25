# ====================================== LIBRARIES ======================================
import torch
import torch.nn as nn

import math
# =======================================================================================


class TimeStepEmbedding(nn.Module):

    '''
        Sinusoidal Timestep Embedding Block.

        Encodes scalar timesteps using sinusoidal positional encoding followed by a learnable MLP projection.

        Parameters
        ----------
        embeddingDimension: int
         Base dimension of the sinusoidal embedding (MUST be even). The final output dimension after the MLP projection is 4 * embeddingDimension.

        Shapes
        ----------
        Input:  
         - t: torch.Tensor of shape [BatchSize, ]. Scalar timesteps (int) representing diffusion step indices. 
        
        Output: 
         - t: torch.Tensor of shape [BatchSize, 4 * embeddingDimension]
    '''

    def __init__(self, embeddingDimension: int) -> None:

        super().__init__()

        if embeddingDimension % 2 != 0:
            raise ValueError(f"As standard, the Sinusoidal Positional Encoding requires an even embedding dimension, but got {embeddingDimension} instead.")

        self.embeddingDimension = embeddingDimension
        self.outputDimension = embeddingDimension * 4

        # Precompute the omegas here to speedup the computation at training/inference time
        halfDim = self.embeddingDimension // 2
        exponents = torch.arange(halfDim, dtype = torch.float32) * -(math.log(10000.0) / halfDim)

        # Registering as a buffer automatically handles device placement (i.e. CPU / GPU)
        self.omegas: torch.Tensor
        self.register_buffer('omegas', torch.exp(exponents))

        # Define the linear layers to be used for the Positional Encoding
        self.layers = nn.Sequential(
                                        nn.Linear(self.embeddingDimension, self.outputDimension),
                                        nn.SiLU(),
                                        nn.Linear(self.outputDimension, self.outputDimension)
                                        
                                    )


    def forward(self, t: torch.Tensor) -> torch.Tensor:

        '''
            t: Time Tensor    ->    [BatchSize]
        '''

        return self.layers(self.SinusoidalPositionalEncoding(t))


    def SinusoidalPositionalEncoding(self, t: torch.Tensor) -> torch.Tensor:

        t = t.float()
        products = t.unsqueeze(1) * self.omegas
        pe = torch.zeros(t.shape[0], self.embeddingDimension, device = t.device)

        pe[:, 0::2] = torch.sin(products)
        pe[:, 1::2] = torch.cos(products)

        return pe
