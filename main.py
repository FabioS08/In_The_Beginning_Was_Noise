# =========================================== LIBRARIES ===========================================
from configs.ddpm_celebahq import config

from datasets import getDataset

from models import getModel
from models.networks.embeddings import TimeStepEmbedding
from models.networks.unet import UNet
from models.ddpm.config import encoderConfig, bottleneckConfig, decoderConfig, initChannels

from tools import *
from utils import *

from torch.utils.data import DataLoader
from torchvision import transforms as T

from pathlib import Path

from rich.traceback import install             # For a better visualization of the errors
install()
# =================================================================================================

device = getDevice()

#* Import the dataset
datasetPath = Path(config.dataset.rootPath).resolve()
transforms = T.Compose([    
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(mean = [0.5] * 3, std = [0.5] * 3)
                        ])

DatasetClass = getDataset(config.dataset.name)
dataset = DatasetClass(datasetPath, download = config.dataset.download, transforms = transforms)


#* Create the dataloader
dataloader = DataLoader(dataset, batch_size = config.training.batchSize, shuffle = True)


#* Create the model
# 1) Build the noise schedule
noiseSchedule = selectNoiseScheduler(config.model.varianceSchedule, **config.model.varianceScheduleParams)

# 2) Build the time embedding
timeEmbedding = TimeStepEmbedding(config.model.timeEmbeddingDimension)

# 3) Build the UNet Network
unetModel = UNet(timeStepEmbedding = timeEmbedding, initChannels = initChannels, 
                 encoder = encoderConfig, bottleneck = bottleneckConfig, decoder = decoderConfig)

# 4) Build the Diffusion Model
ModelClass = getModel(config.model.name)
ddpmModel = ModelClass(unetModel = unetModel, noiseSchedule = noiseSchedule, T = config.model.timesteps) 

#* Define the training parameters
optimizer = buildOptimizer(config.training.optimizer, ddpmModel.parameters(), 
                           config.training.learningRate, **config.training.optimizerParams)
lossFunction = buildLossFunction(config.training.loss)
scheduler = buildScheduler(config.training.scheduler, optimizer, **config.training.schedulerParams)

trainer = DiffusionTrainer(model = ddpmModel, dataloader = dataloader, 
                           optimizer = optimizer, lossFunction = lossFunction,
                           device = device, 
                           gradientAccumulationSteps = config.training.gradientAccumulationSteps,
                           checkpointSavingFrequency = config.training.checkpointSavingFrequency,
                           maxNumCheckpoints = config.training.maxNumCheckpoints,
                           checkpointPathRestart = Path(config.training.checkpointPathRestart) if config.training.checkpointPathRestart else None,
                           ampType = selectAMPType(config.training.ampType), 
                           scheduler = scheduler, 
                           maxGradNorm = config.training.maxGradNorm, 
                           checkpointDirectory = Path(config.training.checkpointDirectory))

trainer.train(epochs = config.training.epochs) 