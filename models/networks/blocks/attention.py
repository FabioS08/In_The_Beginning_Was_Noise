# ====================================== LIBRARIES ======================================
import torch.nn as nn
import torch.nn.functional as F
# =======================================================================================


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
