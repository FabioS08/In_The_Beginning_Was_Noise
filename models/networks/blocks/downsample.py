# ====================================== LIBRARIES ======================================
import torch.nn as nn
# =======================================================================================


class DownSampleBlock(nn.Module):

    '''
        Convolutional downsampling block.

        Applies a 2D convolution with stride > 1 to reduce spatial resolution while increasing/preserving the channel depth.

        Parameters
        ----------

        inChannels: int 
         Number of channels in the input image
        
        outChannels: int
         Number of channels produced by the convolution
        
        kernelSize: int
         Size of the convolving kernel [Default: 3]
        
        stride: int, optional
         Stride of the convolution. [Default: 2]
        
        padding: int, optional
         Padding added to all sides. [Default: 1]

         
        Shapes
        ----------
        Input:  
         - x: torch.Tensor of shape [BatchSize, inChannels, Height, Width]

        Output: 
         - x: torch.Tensor of shape [BatchSize, outChannels, outHeight, outWidth]

        Formula
        ---------    
         - outHeight = floor((H + 2P - (K - 1) - 1) / S + 1)
         
         - outWidth = floor((W + 2P - (K - 1) - 1) / S + 1)

        where K is the kernel size, S the stride, and P the padding.
        '''

    def __init__(self, inChannels: int, outChannels: int, kernelSize: int = 3, stride: int = 2, padding: int = 1):

        super().__init__()

        self.conv = nn.Conv2d(in_channels = inChannels, out_channels = outChannels, kernel_size = kernelSize,
                              stride = stride, padding = padding)


    def forward(self, x):

        return self.conv(x)
