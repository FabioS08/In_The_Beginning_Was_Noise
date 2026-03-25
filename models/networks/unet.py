# ====================================== LIBRARIES ======================================
import torch
import torch.nn as nn
from typing import List

from models.networks.embeddings import TimeStepEmbedding
from models.networks.blocks import ResBlock, MultiHeadAttentionBlock, DownSampleBlock, UpSampleBlock, EncoderItem, BottleneckItem, DecoderItem
# =======================================================================================


class SkipConnectionsError(Exception):
    pass
    

class UNet(nn.Module):

    '''
    U-Net Architecture for Diffusion Models.
    
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
        for item in encoder:

            if item[0] == 'ResNet':
                param = item[1]

                self.encoder.append(ResBlock(inChannels = currentChannels, 
                                             outChannels = param['outChannels'], 
                                             dTimeEmbedding = timeStepEmbedding.outputDimension, 
                                             numGroups = param.get('numGroups', 32), 
                                             dropout = param.get('dropout', 0.1)))
                
                currentChannels = param['outChannels']
                skipChannelStack.append(currentChannels)
                
            elif item[0] == 'DownSample':
                param = item[1]

                self.encoder.append(DownSampleBlock(inChannels = currentChannels, 
                                                    outChannels = param['outChannels'], 
                                                    kernelSize = param.get('kernelSize', 3), 
                                                    stride = param.get('stride', 2), 
                                                    padding = param.get('padding', 1)))
                
                currentChannels = param['outChannels']
                skipChannelStack.append(currentChannels)

            elif item[0] == 'Attention':
                param = item[1]

                self.encoder.append(MultiHeadAttentionBlock(inChannels = currentChannels, 
                                                            numHeads = param['numHeads'], 
                                                            numGroups = param.get('numGroups', 32), 
                                                            zeroInit = param.get('zeroInit', True)))
                
            else:
                raise ValueError(f'Unknown block type "{item[0]}" in the encoder configuration. Expected "ResNet", "DownSample" or "Attention".')


        # 2. Build Bottleneck 
        for item in bottleneck:

            if item[0] == 'ResNet':
                param = item[1]

                self.bottleneck.append(ResBlock(inChannels = currentChannels, 
                                                outChannels = param['outChannels'], 
                                                dTimeEmbedding = timeStepEmbedding.outputDimension, 
                                                numGroups = param.get('numGroups', 32), 
                                                dropout = param.get('dropout', 0.1)))
                
                currentChannels = param['outChannels']
                
            elif item[0] == 'Attention':
                param = item[1]

                self.bottleneck.append(MultiHeadAttentionBlock(inChannels = currentChannels, 
                                                               numHeads = param['numHeads'], 
                                                               numGroups = param.get('numGroups', 32), 
                                                               zeroInit = param.get('zeroInit', True)))
                
            else:
                raise ValueError(f'Unknown block type "{item[0]}" in the bottleneck configuration. Expected "ResNet" or "Attention".')


        # 3. Build Decoder
        for item in decoder:

            if item[0] == 'ResNet':
                param = item[1]

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

            elif item[0] == 'UpSample':
                param = item[1]

                self.decoder.append(UpSampleBlock(inChannels = currentChannels, 
                                                  outChannels = param['outChannels'], 
                                                  kernelSize = param.get('kernelSize', 3), 
                                                  stride = param.get('stride', 1), 
                                                  padding = param.get('padding', 1)))
                
                currentChannels = param['outChannels']

            elif item[0] == 'Attention':
                param = item[1]

                self.decoder.append(MultiHeadAttentionBlock(inChannels = currentChannels, 
                                                            numHeads = param['numHeads'], 
                                                            numGroups = param.get('numGroups', 32), 
                                                            zeroInit = param.get('zeroInit', True)))
            
            else:
                raise ValueError(f'Unknown block type "{item[0]}" in the decoder configuration. Expected "ResNet", "UpSample" or "Attention".')


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

                try:

                    skip = skips.pop()

                except:
                    
                    raise SkipConnectionsError(f"Attempt to retrieve a skip connection from an empy list. This likely means that the encoder and decoder configurations are not properly aligned (Recall: if N ResBlocks/AttentionBlocks in the encoder, then N + 1 ResBlocks/AttentionBlocks in the decoder).")

                h = torch.cat([h, skip], dim = 1)
                h = block(h, tEmbedding)

            elif isinstance(block, MultiHeadAttentionBlock) or isinstance(block, UpSampleBlock):
                h = block(h)

        if skips:

            raise SkipConnectionsError(f"Skip connection stack is not empty at the end of the forward pass. Remaining skips: {len(skips)}. This likely means that the encoder and decoder configurations are not properly aligned (Recall: if N ResBlocks/AttentionBlocks in the encoder, then N + 1 ResBlocks/AttentionBlocks in the decoder).")

        # Return to a 3-channel element    
        h = self.exit(h)

        return h