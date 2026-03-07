# ====================================== LIBRARIES ======================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal, Tuple, TypedDict, Union
from typing_extensions import NotRequired
import math
# =======================================================================================


#* Implement the Time Step Embedding to use when in input it is provided the image and the timestep of the noise applicated to that image
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


#* Implement the Residual Block to use in the architecture
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


#* Implement the Attention Block to use in the latent space
class MultiHeadAttentionBlock(nn.Module):

    '''
        Spatial Multi-Head Self-Attention Block.

        Applies self-attention over spatial positions of a feature map. Each spatial location attends to every other location in the feature map.

        Parameters
        ----------
        numHeads: int 
         Number of attention heads.
        
        inChannels: int
         Number of input feature channels.
        
        numGroups: int, optional 
         Number of groups used in GroupNorm. [Default: 32]
        
        zeroInit: bool, optional
         If True, initializes the output projection weights to zero to stabilize early training. [Default: True]

        Shapes
        ----------
        Input:  
         - x: torch.Tensor of shape [BatchSize, inChannels, Height, Width]

        Output:
         - x: torch.Tensor of shape [BatchSize, inChannels, Height, Width]        
    '''

    def __init__(self, numHeads: int, inChannels: int, numGroups: int = 32, zeroInit: bool = True):

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

        # Output Projection
        self.O = nn.Conv2d(in_channels = inChannels, out_channels = inChannels, kernel_size = 1)

        # At the beginning the Zero-Initialization Trick is used to ensure that the attention block acts as an identity function (i.e. the variance of the network is kept stable at the start of the training)
        if zeroInit:

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


#* Implement the DownSample Block to reduce the spatial resolution
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


#* Implement the UpSample Block to increase the spatial resolution
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


#* Define the Block Parameters Classes (just for type hinting and readability)
class ResBlockParams(TypedDict):
    outChannels: int
    inChannels: NotRequired[int]
    numGroups: NotRequired[int]
    dropout: NotRequired[float]

class AttentionBlockParams(TypedDict):
    numHeads: int
    inChannels: NotRequired[int]
    numGroups: NotRequired[int]
    zeroInit: NotRequired[bool]

class UpDownBlockParams(TypedDict):
    outChannels: int
    inChannels: NotRequired[int]
    kernelSize: NotRequired[int]
    stride: NotRequired[int]
    padding: NotRequired[int]


EncoderItem = Union[Tuple[Literal["ResNet"], ResBlockParams], 
                    Tuple[Literal["DownSample"], UpDownBlockParams],
                    Tuple[Literal["Attention"], AttentionBlockParams]] 

BottleneckItem = Union[Tuple[Literal["ResNet"], ResBlockParams], 
                       Tuple[Literal["Attention"], AttentionBlockParams]]

DecoderItem = Union[Tuple[Literal["ResNet"], ResBlockParams], 
                    Tuple[Literal["UpSample"], UpDownBlockParams],
                    Tuple[Literal["Attention"], AttentionBlockParams]]


class SkipConnectionsError(Exception):
    pass
    

#* Implement the Encoder-Decoder Architecture (i.e. shrink the image to a latent space and, then, upsample it)
class UNet(nn.Module):

    '''
    U-Net Architecture for the DDDPM Model.
    
    Compresses the input image to a latent space through a series of ResBlocks/AttentionBlocks/DownSampleBlocks (Encoder) and, then, reconstruct the final image through a series of ResBlocks/AttentionBlocks/UpSampleBlocks (Decoder).

    Parameters
    ---------- 
    timeStepEmbedding: TimeStepEmbedding 
     The embedding module used to convert the raw timestep input into a vector to be injected into the ResBlocks
    
    initChannels: int
     The number of channels used as starting point of the Encoder.

    encoder: List[EncoderItem]
     The list containing the blocks to be used in the encoder. Each item in the list is a tuple of the form (BlockType, BlockParameters) where:
                               - BlockType is a string that can be either "ResNet", "DownSample" or "Attention"
                               - BlockParameters is a dictionary containing the parameters of that block (the only mandatory ones are outChannels for 'ResNet' and 'DownSample' and numHeads for 'Attention' - Look at the documentation for the list about the optional ones)

    bottleneck: List[BottleneckItem] 
     The list containing the blocks to be used in the bottleneck. Each item in the list is a tuple of the form (BlockType, BlockParameters) where:
                               - BlockType is a string that can be either "ResNet" or "Attention"
                               - BlockParameters is a dictionary containing the parameters of that block (the only mandatory ones are outChannels for 'ResNet' and numHeads for 'Attention')

    decoder: List[DecoderItem]
     The list containing the blocks to be used in the decoder. Each item in the list is a tuple of the form (BlockType, BlockParameters) where:
                               - BlockType is a string that can be either "ResNet", "UpSample" or "Attention"
                               - BlockParameters is a dictionary containing the parameters of that block (the only mandatory ones are outChannels for 'ResNet' and 'UpSample' and numHeads for 'Attention')

    '''

    def __init__(self, timeStepEmbedding: TimeStepEmbedding, initChannels: int, encoder: List[EncoderItem], bottleneck: List[BottleneckItem], decoder: List[DecoderItem]) -> None:

        super().__init__()

        self.timeStepEmbedding = timeStepEmbedding
        self.entryConv = nn.Conv2d(in_channels = 3, out_channels = initChannels, kernel_size = 3, padding = 1, stride = 1)
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.exit = nn.ModuleList()

        skipChannelStack = [initChannels]   # We will use this to track the output channels of every encoder block
        currentChannels = initChannels

        # 1. Build Encoder
        for blockType, param in encoder:

            if blockType == 'ResNet':

                self.encoder.append(ResBlock(inChannels = currentChannels, 
                                             outChannels = param['outChannels'], 
                                             dTimeEmbedding = timeStepEmbedding.outputDimension, 
                                             numGroups = param.get('numGroups', 32), 
                                             dropout = param.get('dropout', 0.1)))
                
                currentChannels = param['outChannels']
                
            elif blockType == 'DownSample':

                self.encoder.append(DownSampleBlock(inChannels = currentChannels, 
                                                    outChannels = param['outChannels'], 
                                                    kernelSize = param.get('kernelSize', 3), 
                                                    stride = param.get('stride', 2), 
                                                    padding = param.get('padding', 1)))
                
                currentChannels = param['outChannels']

            elif blockType == 'Attention':
                self.encoder.append(MultiHeadAttentionBlock(inChannels = currentChannels, 
                                                            numHeads = param['numHeads'], 
                                                            numGroups = param.get('numGroups', 32), 
                                                            zeroInit = param.get('zeroInit', True)))
                
            else:
                raise ValueError(f'Unknown block type "{blockType}" in the encoder configuration. Expected "ResNet", "DownSample" or "Attention".')
            
            # Save the output channel dimension for the decoder to know about
            skipChannelStack.append(currentChannels)

        # 2. Build Bottleneck 
        for blockType, param in bottleneck:

            if blockType == 'ResNet':

                self.bottleneck.append(ResBlock(inChannels = currentChannels, 
                                                outChannels = param['outChannels'], 
                                                dTimeEmbedding = timeStepEmbedding.outputDimension, 
                                                numGroups = param.get('numGroups', 32), 
                                                dropout = param.get('dropout', 0.1)))
                
                currentChannels = param['outChannels']
                
            elif blockType == 'Attention':

                self.bottleneck.append(MultiHeadAttentionBlock(inChannels = currentChannels, 
                                                               numHeads = param['numHeads'], 
                                                               numGroups = param.get('numGroups', 32), 
                                                               zeroInit = param.get('zeroInit', True)))
                
            else:
                raise ValueError(f'Unknown block type "{blockType}" in the bottleneck configuration. Expected "ResNet" or "Attention".')



        # 3. Build Decoder
        for blockType, param in decoder:

            if blockType == 'ResNet':

                # Pop the expected channels from the skip connection
                skipChannels = skipChannelStack.pop()
                
                # The real input channels for the decoder block is current + skip
                actualChannels = currentChannels + skipChannels
                
                self.decoder.append(ResBlock(inChannels = actualChannels, 
                                             outChannels = param['outChannels'], 
                                             dTimeEmbedding = timeStepEmbedding.outputDimension, 
                                             numGroups = param.get('numGroups', 32), 
                                             dropout = param.get('dropout', 0.1)))
                currentChannels = param['outChannels']

            elif blockType == 'UpSample':

                self.decoder.append(UpSampleBlock(inChannels = currentChannels, 
                                                  outChannels = param['outChannels'], 
                                                  kernelSize = param.get('kernelSize', 3), 
                                                  stride = param.get('stride', 1), 
                                                  padding = param.get('padding', 1)))
                
                currentChannels = param['outChannels']

            elif blockType == 'Attention':

                self.decoder.append(MultiHeadAttentionBlock(inChannels = currentChannels, 
                                                            numHeads = param['numHeads'], 
                                                            numGroups = param.get('numGroups', 32), 
                                                            zeroInit = param.get('zeroInit', True)))
            
            else:
                raise ValueError(f'Unknown block type "{blockType}" in the decoder configuration. Expected "ResNet", "UpSample" or "Attention".')


        # 4. Build Exit
        self.exit = nn.Sequential(
                                    nn.GroupNorm(num_channels = currentChannels, num_groups = 32), 
                                    nn.SiLU(),
                                    nn.Conv2d(in_channels = currentChannels, out_channels = 3, kernel_size = 3, padding = 1, stride = 1)
                                )
                                             

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        tEmbedding = self.timeStepEmbedding(t)

        h = self.entryConv(x)
        skips = [h]

        # Apply the Encoder's Blocks and save the residuals
        for block in self.encoder:

            if isinstance(block, ResBlock):

                h = block(h, tEmbedding)
                skips.append(h)                 # Push the ResBlock output
                
            elif isinstance(block, MultiHeadAttentionBlock):

                h = block(h)
                skips[-1] = h                   # OVERWRITE the last skip connection! Do not append.
                
            elif isinstance(block, DownSampleBlock):

                h = block(h)
                skips.append(h)                 # Push the DownSample output


        # Work on the bottleneck space
        for block in self.bottleneck:
            
            if isinstance(block, ResBlock):
                h = block(h, tEmbedding)
            
            else:
                h = block(h)


        # Apply the Decoder's Blocks and concatenate the skip connections
        for block in self.decoder:

            if isinstance(block, ResBlock):

                skip = skips.pop()           
                h = torch.cat([h, skip], dim = 1)
                h = block(h, tEmbedding)

            elif isinstance(block, MultiHeadAttentionBlock) or isinstance(block, UpSampleBlock):
                h = block(h)

        if skips:

            raise SkipConnectionsError(f"Skip connection stack is not empty at the end of the forward pass. Remaining skips: {len(skips)}. This likely means that the encoder and decoder configurations are not properly aligned (Recall: if N ResBlocks/AttentionBlocks in the encoder, then N + 1 ResBlocks/AttentionBlocks in the decoder).")

        # Return to a 3-channel element    
        h = self.exit(h)

        return h