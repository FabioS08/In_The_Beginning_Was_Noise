# ====================================== LIBRARIES ======================================
import torch
import torch.nn as nn
import torch.nn.functional as F
# =======================================================================================


class ResBlock(nn.Module):

    '''
        Residual convolutional block with timestep conditioning.

        Applies two convolutional layers with Group Normalization and injects a projected timestep embedding between them.

        Parameters
        ----------
        inChannels: int 
         Number of input channels.
        
        outChannels: int
         Number of output channels.
        
        dTimeEmbedding: int 
         Dimension of the timestep embedding.
        
        numGroups: int, optional 
         Number of groups used in GroupNorm. [Default: 32]
        
        dropout: float, optional 
         Dropout probability applied before the second convolution. [Default: 0.1]

        Shapes
        ----------
        Input:  
         - x: torch.Tensor of shape [BatchSize, inChannels, Height, Width]
         - t: torch.Tensor of shape [BatchSize, dTimeEmbedding]

        Output:   
         - x: torch.Tensor of shape [BatchSize, outChannels, Height, Width]
    '''

    def __init__(self, inChannels: int, outChannels: int, dTimeEmbedding: int, numGroups: int = 32, dropout: float = 0.1) -> None:

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
