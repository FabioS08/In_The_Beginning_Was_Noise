from models.UNet import EncoderItem, DecoderItem, BottleneckItem
from typing import List

# =============================================== Configuration Example ===============================================


#* RULES:
#*          1. No downsample block as last element of the encoder
#*          2. No upsample block as last element of the decoder
#*          3. If the encoder had N ResBlocks per level, the decoder has N + 1 ResBlocks per level 
#*             (i.e. the features extracted with the ResBlocks + the Downsample Output)
#*             Note that an Attention block will decrease the count of 1 (i.e. 3 ResBlocks in the encoder will correspond to 4 #*             ResBlocks in the decoder, but if one of those 3 ResBlocks is replaced with an Attention block, then we will have 3 #*             ResBlocks in the decoder)
#*          4. The encoder must have an additional ResBlock as last element to consume the 'entry convolution' skip connection


# =====================================================================================================================
# Below there is a model of each block type:
# 
# - ('ResNet', {outChannels: int, inChannels: Optional[int], numGroups: Optional[int], dropout: Optional[float]})
#
# - ('Attention', {numHeads: int, inChannels: Optional[int], numGroups: Optional[int], zeroInit: Optional[bool]})
#
# - ('DownSample', {outChannels: int, inChannels: Optional[int], kernelSize: Optional[int], stride: Optional[int], 
#                   padding: Optional[int]})
#
# - ('UpSample', {SAME OF DOWNSAMPLE})
# =====================================================================================================================

initChannels = 128

encoderConfig: List[EncoderItem] = [

    # Level 0 (256 x 256 and C = 128)
    ("ResNet", {"outChannels": 128}),       # Saves Skip  1
    ("ResNet", {"outChannels": 128}),       # Saves Skip  2
    ("DownSample", {"outChannels": 128}),   # Saves Skip  3

    # Level 1 (128 x 128 and C = 128)
    ("ResNet", {"outChannels": 128}),       # Saves Skip  4
    ("ResNet", {"outChannels": 128}),       # Saves Skip  5
    ("DownSample", {"outChannels": 256}),   # Saves Skip  6

    # Level 2 (64 x 64 and C = 256)
    ("ResNet", {"outChannels": 256}),       # Saves Skip  7
    ("ResNet", {"outChannels": 256}),       # Saves Skip  8
    ("DownSample", {"outChannels": 256}),   # Saves Skip  9

    # Level 3 (32 x 32 and C = 256)
    ("ResNet", {"outChannels": 256}),       # Saves Skip 10
    ("ResNet", {"outChannels": 256}),       # Saves Skip 11
    ("DownSample", {"outChannels": 256}),   # Saves Skip 12

    # Level 4 (16 x 16 and C = 512)
    ("ResNet", {"outChannels": 512}),       # Saves Skip 13
    ("Attention", {"numHeads": 1}),         # Overwrite Skip 13
    ("ResNet", {"outChannels": 512}),       # Saves Skip 15
    ("DownSample", {"outChannels": 512}),   # Saves Skip 16

    # Level 5 (8 x 8 and C = 512)
    ("ResNet", {"outChannels": 512}),       # Saves Skip 17
    ("ResNet", {"outChannels": 512}),       # Saves Skip 18
    
    # Notice: NO DownSample block here! The encoder just stops.
]


bottleneckConfig: List[BottleneckItem] = [

    # No Skip Connections here!
    ("ResNet", {"outChannels": 512}),       
    ("Attention", {"numHeads": 1}),         
    ("ResNet", {"outChannels": 512}),       

]


decoderConfig: List[DecoderItem] = [

    # Level 1 (8 x 8 and C = 512)
    ("ResNet", {"outChannels": 512}),       # Consumes Skip 18
    ("ResNet", {"outChannels": 512}),       # Consumes Skip 17
    ("ResNet", {"outChannels": 512}),       # Consumes Skip 16
    ("UpSample", {"outChannels": 512}),     

    # Level 2 (16 x 16 and C = 512)
    ("ResNet", {"outChannels": 512}),       # Consumes Skip 15
    ("Attention", {"numHeads": 1}), 
    ("ResNet", {"outChannels": 512}),       # Consumes Skip 14
    ("ResNet", {"outChannels": 512}),       # Consumes Skip 13
    ("UpSample", {"outChannels": 512}),  

    # Level 3 (32 x 32 and C = 256)
    ("ResNet", {"outChannels": 256}),       # Consumes Skip 12
    ("ResNet", {"outChannels": 256}),       # Consumes Skip 11
    ("ResNet", {"outChannels": 256}),       # Consumes Skip 10
    ("UpSample", {"outChannels": 256}),  

    # Level 4 (64 x 64 and C = 256)
    ("ResNet", {"outChannels": 256}),       # Consumes Skip  9
    ("ResNet", {"outChannels": 256}),       # Consumes Skip  8
    ("ResNet", {"outChannels": 256}),       # Consumes Skip 7
    ("UpSample", {"outChannels": 256}),  

    # Level 5 (128 x 128 and C = 128)
    ("ResNet", {"outChannels": 128}),       # Consumes Skip  6
    ("ResNet", {"outChannels": 128}),       # Consumes Skip  5
    ("ResNet", {"outChannels": 128}),       # Consumes Skip  4
    ("UpSample", {"outChannels": 128}),  

    # Level 6 (256 x 256 and C = 128)
    ("ResNet", {"outChannels": 128}),       # Consumes Skip  3
    ("ResNet", {"outChannels": 128}),       # Consumes Skip  2

    ("ResNet", {"outChannels": 128}),       # Consumes Skip  1 (the one of the entry convolution - from 3 to initChannels)

    # Notice: NO UpSample block here! 
]

