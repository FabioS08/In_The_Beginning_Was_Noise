# Re-export all blocks and type definitions for convenient imports
from models.networks.blocks.resblock import ResBlock
from models.networks.blocks.attention import MultiHeadAttentionBlock
from models.networks.blocks.downsample import DownSampleBlock
from models.networks.blocks.upsample import UpSampleBlock

from typing import Literal, Tuple, Union
from typing_extensions import NotRequired
from typing import TypedDict


# ============================== Block Parameter Types ==============================
# Used for type hinting and readability in UNet configuration files

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


# ============================== Configuration Item Types ==============================

EncoderItem = Union[Tuple[Literal["ResNet"], ResBlockParams], 
                    Tuple[Literal["DownSample"], UpDownBlockParams],
                    Tuple[Literal["Attention"], AttentionBlockParams]] 

BottleneckItem = Union[Tuple[Literal["ResNet"], ResBlockParams], 
                       Tuple[Literal["Attention"], AttentionBlockParams]]

DecoderItem = Union[Tuple[Literal["ResNet"], ResBlockParams], 
                    Tuple[Literal["UpSample"], UpDownBlockParams],
                    Tuple[Literal["Attention"], AttentionBlockParams]]