import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeStepEmbedding(nn.Module):

    def __init__(self, embeddingDimension: int) -> None:

        '''
            :param embeddingDimension: The dimension of the embedding to be generated
            :type embeddingDimension: int
        '''

        super().__init__()

        if embeddingDimension % 2 != 0:
            raise ValueError(f"As standard, the Sinusoidal Positional Encoding requires an even embedding dimension, but got {embeddingDimension} instead.")

        self.embeddingDimension = embeddingDimension

        # Define the linear layers to be used for the Positional Encoding
        self.layers = nn.Sequential(
                                        nn.Linear(self.embeddingDimension, self.embeddingDimension * 4),
                                        nn.SiLU(),
                                        nn.Linear(self.embeddingDimension * 4, self.embeddingDimension * 4)
                                        
                                    )


    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.layers(self.SinusoidalPositionalEncoding(t))


    def SinusoidalPositionalEncoding(self, t: torch.Tensor) -> torch.Tensor:

        pe = torch.zeros(t.shape[0], self.embeddingDimension)
        exponents = -(2 * torch.arange(0, self.embeddingDimension // 2)) / self.embeddingDimension
        omegas = torch.pow(10000, exponents)
        products = t.unsqueeze(1).expand(-1, self.embeddingDimension // 2) * omegas

        pe[:, 0::2] = torch.sin(products)
        pe[:, 1::2] = torch.cos(products)

        return pe



class UNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x