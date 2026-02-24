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

        # Precompute the omegas here to speedup the computation at training/inference time
        halfDim = self.embeddingDimension // 2
        exponents = torch.arange(halfDim, dtype = torch.float32) * -(math.log(10000.0) / halfDim)

        # Registering as a buffer automatically handles device placement (i.e. CPU / GPU)
        self.omegas: torch.Tensor
        self.register_buffer('omegas', torch.exp(exponents))

        # Define the linear layers to be used for the Positional Encoding
        self.layers = nn.Sequential(
                                        nn.Linear(self.embeddingDimension, self.embeddingDimension * 4),
                                        nn.SiLU(),
                                        nn.Linear(self.embeddingDimension * 4, self.embeddingDimension * 4)
                                        
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


#* Implement the Attention Block
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, numHeads: int, inChannels: int, numGroups: int = 32):

        '''
            :param numHeads: The number of channels of the tensor in input
            :type numHeads: int

            :param inChannels: The number of channels of the tensor in input
            :type inChannels: int

            :param numGroups: The number of channels of the tensor in input
            :type numGroups: int
        
        '''

        super().__init__()

        if inChannels % numHeads != 0:
            raise ValueError(f'The number of channels must be divisible by the number of attention head: got {inChannels} (inChannels) / {numHeads} (numHeads) = {inChannels / numHeads}')
        
        self.numHeads = numHeads
        self.hiddenDimension = inChannels // numHeads   # The typical notation is d_k

        self.norm = nn.GroupNorm(num_channels = inChannels, num_groups = numGroups)

        # Perform the projections to generate the Query, Key and Value Matrices
        self.Q = nn.Conv2d(in_channels = inChannels, out_channels = inChannels, kernel_size = 1)
        self.K = nn.Conv2d(in_channels = inChannels, out_channels = inChannels, kernel_size = 1)
        self.V = nn.Conv2d(in_channels = inChannels, out_channels = inChannels, kernel_size = 1)

        # Output Projection - At the beginning the Zero-Initialization Trick is used to ensure that the attention block acts as an identity function (i.e. the variance of the network is kept stable at the start of the training)
        self.O = nn.Conv2d(in_channels = inChannels, out_channels = inChannels, kernel_size = 1)
        nn.init.zeros_(self.O.weight)
        if self.O.bias is not None:
            nn.init.zeros_(self.O.bias)


    def forward(self, x):
        
        '''
            x: Input Image Tensor       -> [BatchSize, inChannels, H, W]
        '''

        B, C, H, W = x.shape
        L = H * W 

        # Normalize the input 
        h = self.norm(x)

        # Create the Q, K, V Matrices   ->  [BachSize, NumHeads, L, hiddenDimension]
        Q = self.Q(h).reshape(B, self.numHeads, self.hiddenDimension, L).transpose(2, 3)
        K = self.K(h).reshape(B, self.numHeads, self.hiddenDimension, L).transpose(2, 3)
        V = self.V(h).reshape(B, self.numHeads, self.hiddenDimension, L).transpose(2, 3)

        # Compute the Attention Output   ->  [BachSize, NumHeads, L, hiddenDimension]
        out = F.scaled_dot_product_attention(Q, K, V)
    
        # Reshape the output   ->  [BachSize, inChannels, H, W]
        out = out.transpose(2, 3).reshape(B, C, H, W)
        out = self.O(out)
        
        return x + out








class UNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x