# =========================================== LIBRARIES ===========================================
from datasets import getDataset
from models import buildModel

from tools import *
from utils import *

from torch.utils.data import DataLoader
from torchvision import transforms as T

from pathlib import Path

from rich.traceback import install             # For a better visualization of the errors
install()
# =================================================================================================


#* Parse command-line arguments
config = TrainCLI().parse()

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

#* Build the model from the configuration object
model = buildModel(config)

#* Define the training parameters
optimizer = buildOptimizer(config.training.optimizer, model.parameters(), 
                           config.training.learningRate, **config.training.optimizerParams)
lossFunction = buildLossFunction(config.training.loss)
scheduler = buildScheduler(config.training.scheduler, optimizer, **config.training.schedulerParams)

trainer = DiffusionTrainer(model = model, dataloader = dataloader, 
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