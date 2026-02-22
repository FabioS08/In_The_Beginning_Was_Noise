import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#* Implement the Time Step Embedding to use when in input it is provided the image and the timestep of the noise applicated to that image
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

        device = t.device

        pe = torch.zeros(t.shape[0], self.embeddingDimension, device = device)
        halfDim = self.embeddingDimension // 2
        exponents = torch.arange(halfDim, device = device) * -(math.log(10000.0) / halfDim)
        omegas = torch.exp(exponents)
        products = t.unsqueeze(1) * omegas

        pe[:, 0::2] = torch.sin(products)
        pe[:, 1::2] = torch.cos(products)

        return pe


#* Implement the ResNet Block to use in the architecture
class ResBlock(nn.Module):

    def __init__(self, inChannels: int, outChannels: int, dTimeEmbedding: int, numGroups: int = 32, dropout: float = 0.1) -> None:

        '''
            :param inChannels: The number of channels of the tensor in input
            :type inChannels: int

            :param outChannels: The number of channels of the tensor in output
            :type outChannels: int

            :param dTimeEmbedding: The embedding dimension of the encoded time vector
            :type dTimeEmbedding: int

            :param numGroups: The number of normalization groups to use in the 'GroupNorm' Layer
            :type numGroups: int

            :param dropout: The probability of dropout to address co-adaptation issues
            :type dropout: float
        '''

        super().__init__()

        if inChannels % numGroups != 0 or outChannels % numGroups != 0:
            raise ValueError(f'Both input channels (inChannels = {inChannels}) and output channels (outChannels = {outChannels}) must be divisible by the number of normalization groups (numGroups = {numGroups}).')

        # Define the elements of the first convolution
        self.norm1 = nn.GroupNorm(num_groups = numGroups, num_channels = inChannels)
        self.conv1 = nn.Conv2d(in_channels = inChannels, out_channels = outChannels, kernel_size = 3, padding = 1)

        # Define the layer to match the time embedding dimension to the current dimension
        self.timeProjection = nn.Sequential(
                                                nn.SiLU(),
                                                nn.Linear(in_features = dTimeEmbedding, out_features = outChannels)            
                                            )
        
        # Define the elements of the second convolution
        self.norm2 = nn.GroupNorm(num_groups = numGroups, num_channels = outChannels)
        self.dropout2 = nn.Dropout(p = dropout)
        self.conv2 = nn.Conv2d(in_channels = outChannels, out_channels = outChannels, kernel_size = 3, padding = 1)

        # Handle the case of channel downsample/upsample
        if inChannels != outChannels:
            self.reshape = nn.Conv2d(in_channels = inChannels, out_channels = outChannels, kernel_size = 1)

        else:
            self.reshape = nn.Identity()


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        '''
            x: Input Image Tensor       -> [BatchSize, inChannels, H, W]
            t: Time Embedding Tensor    -> [BatchSize, dTimeEmbedding]
        '''

        # Apply the first convolutional layer
        convX = self.conv1(F.silu(self.norm1(x)))

        # Time Injection
        projectedT = self.timeProjection(t)
        timedConvX = convX + projectedT[:, :, None, None]

        # Apply the second convolutional layer
        out = self.conv2(self.dropout2(F.silu(self.norm2(timedConvX))))

        return out + self.reshape(x)











class UNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x