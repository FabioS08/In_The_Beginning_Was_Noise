# ====================================== LIBRARIES ======================================
from datasets.CelebAHQ_Dataset import CelebAHQ
from models.UNet import TimeStepEmbedding

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pathlib

# For a better visualization of the errors
from rich.traceback import install
install()
# =======================================================================================


#* Import the dataset
datasetPath = pathlib.Path('/Users/fabioschiliro/Desktop/DDPM/datasets')
transforms = T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean = [0.5] * 3, std = [0.5] * 3)
                        ])
dataset = CelebAHQ(datasetPath, download = True, transforms = transforms)

#* Create the dataloader
batchSize = 64
dataloader = DataLoader(dataset, batch_size = batchSize, shuffle = True)

#* Create the model
a = TimeStepEmbedding(128)


print(a(torch.tensor([1, 2, 3])))