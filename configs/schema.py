# ====================================== LIBRARIES ======================================
from dataclasses import dataclass, field
# =======================================================================================

@dataclass
class ModelConfig:

    '''
        Configuration for the diffusion model
    '''

    name: str = "ddpm"
    timesteps: int = 1000
    varianceSchedule: str = "linear"
    varianceScheduleParams: dict = field(default_factory = lambda: {"betaStart": 1e-4, "betaEnd": 0.02})
    timeEmbeddingDimension: int = 128


@dataclass
class DatasetConfig:

    '''
        Configuration for the dataset
    '''

    name: str = "celebahq"
    rootPath: str = "./datasets"
    download: bool = False


@dataclass
class TrainingConfig:

    '''
        Configuration for the training
    '''

    batchSize: int = 16
    epochs: int = 100
    learningRate: float = 0.0001
    optimizer: str = "adamw"
    optimizerParams: dict = field(default_factory = dict)
    loss: str = "mse"
    scheduler: str = ''
    schedulerParams: dict = field(default_factory = dict)
    gradientAccumulationSteps: int = 1
    ampType: str = ''
    checkpointSavingFrequency: int = 5
    maxNumCheckpoints: int = 1
    checkpointPathRestart: str = ''
    maxGradNorm: float = 10.0
    checkpointDirectory: str = ''


@dataclass
class Config:
    
    '''
        Top-level configuration dataclass
    '''

    model: ModelConfig = field(default_factory = ModelConfig)
    dataset: DatasetConfig = field(default_factory = DatasetConfig)
    training: TrainingConfig = field(default_factory = TrainingConfig)
