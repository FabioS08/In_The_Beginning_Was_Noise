# ====================================== LIBRARIES ======================================
import torch.nn as nn
import torch.nn.functional as F
# =======================================================================================


class UpSampleBlock(nn.Module):

    '''
        Convolutional UpSampling block.

        Performs spatial upsampling using nearest-neighbor interpolation followed by a 2D convolution. This design avoids checkerboard artifacts commonly introduced by transposed convolutions.

        Parameters
        ----------

        inChannels: int 
         Number of channels in the input image
        
        outChannels: int
         Number of channels produced by the convolution
        
        kernelSize: int
         Size of the convolving kernel [Default: 3]
        
        stride: int, optional
         Stride of the convolution. [Default: 1]
        
        padding: int, optional
         Padding added to all sides. [Default: 1]

        Shapes
        ----------
        Input:  
         - x: torch.Tensor of shape [BatchSize, inChannels, Height, Width]

        Output: 
         - x: torch.Tensor of shape [BatchSize, outChannels, outHeight, outWidth]

        Upsampling
        ----------
        The Spatial Resolution is first doubled via nearest-neighbor interpolation such that:

            H' = 2H
            W' = 2W

        After interpolation, the convolution produces:
         - outHeight = floor((H' + 2P - (K - 1) - 1) / S + 1)
         
         - outWidth = floor((W' + 2P - (K - 1) - 1) / S + 1)

        where K is the kernel size, S the stride, and P the padding.
    '''

    def __init__(self, inChannels: int, outChannels: int, kernelSize: int = 3, stride: int = 1, padding: int = 1):

        super().__init__()

        self.conv = nn.Conv2d(in_channels = inChannels, out_channels = outChannels, kernel_size = kernelSize,
                              stride = stride, padding = padding)


    def forward(self, x):

        return self.conv(F.interpolate(input = x, mode = 'nearest', scale_factor = 2))
