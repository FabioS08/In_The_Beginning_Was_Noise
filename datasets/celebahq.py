# ====================================== LIBRARIES ======================================
from datasets import registerDataset

import torchvision.transforms as T
from torch.utils.data import Dataset
from zipfile import ZipFile
from typing import Optional
from rich import print
import PIL.ImageShow
import PIL.Image
import pathlib
import gdown
import PIL
# =======================================================================================


@registerDataset('celebahq')
class CelebAHQ(Dataset):

    '''
        CelebA-HQ Dataset

        It handles the downloading, extraction and loading phases of the dataset

        Parameters
        ----------
        rootPath: pathlib.Path
         The path to the folder where the dataset will be saved.

        download: bool
         Whether the dataset should be downlaoded if it is not already present in the rootPath.

        transforms: Optional[T.Compose]
         The transforms to apply to each image in the dataset. [Default: None]

    '''

    def __init__(self, rootPath: pathlib.Path, download: bool = False, transforms: Optional[T.Compose] = None):

        self.rootPath = rootPath
        self.transforms = transforms
        self.fileID = "1jlQ8umhpJo8lVgC9q4_1q_t_Frv1kZ3f"
        self.URL = f"https://drive.google.com/uc?id={self.fileID}"
        self.zipPath = rootPath / 'CelebAHQ.zip'
        self.destPath = rootPath / "CelebA-HQ"

        # Print some info about the dataset
        self.printDatasetInfo()

        # If the dataset has not been downlaoded yet, do it and extract the files
        if download and (not self.destPath.exists() or not any(self.destPath.iterdir())):

            try:
                    # Download the zip file from Google Drive
                    print('[green][*] Downloading the dataset...[/green]')
                    gdown.download(self.URL, str(self.zipPath))
                    print('[green][✓] Download completed![/green]\n')

                    # Extract the zip file
                    print('[green][*] Extracting the dataset...[/green]')
                    with ZipFile(self.zipPath, 'r') as f:
                        f.extractall(self.rootPath)
                    print('[green][✓] Extraction completed![/green]\n')
                    
                    # Rename the extracted folder for a better structure
                    (pathlib.Path(self.rootPath) / "images").rename(self.destPath)
                    print(f"[green][✓] Dataset correctly saved in: '{self.destPath}'[/green]")

            finally:

                # Clean up the zip file after extraction
                if self.zipPath.exists():
                    self.zipPath.unlink()

        self.files = sorted(self.destPath.iterdir(), key = lambda x: int(x.stem))
            


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):

        imgPath = self.files[idx]
        img = PIL.Image.open(pathlib.Path(self.destPath / imgPath)).convert("RGB")
        
        if self.transforms:
             return self.transforms(img)
        
        return img
        

    def printDatasetInfo(self):

        l = max(len(str(self.rootPath)), len(self.URL), len(str(self.zipPath))) // 2

        print('[dim]=[/dim]' * l + ' [reverse blue]DATASET INFO[/reverse blue] ' + '[dim]=[/dim]' * l)
        print(f'[bold]Root Path[/bold]: [italic]{self.rootPath}[/italic]')
        print(f'[bold]Zip File[/bold]: [italic]{self.zipPath}[/italic]')
        print(f'[bold]URL[/bold]: [italic]{self.URL}[/italic]')
        print('[dim]=[/dim]' * (2 * l + 14) + '\n')


if __name__ == "__main__":

    datasetPath = pathlib.Path('/Users/fabioschiliro/Documents/Projects/DDPM/datasets')
    transforms = T.Compose([
                                T.ToTensor(),
                                T.Normalize(mean = [0.5] * 3, std = [0.5] * 3)
                            ])
          
    dataset = CelebAHQ(datasetPath, download = True, transforms = transforms)
    index = 200

    labelWidth = 35
    print(f'{"Number of images in the dataset:":<{labelWidth}} {len(dataset)}')
    print(f'{"Image Size:":<{labelWidth}} {dataset[0].shape}')
    print(f'{"Index of visualized image:":<{labelWidth}} {index}')

    # Denormalize the image for visualization
    img = (dataset[index] + 1) / 2

    PIL.ImageShow.show(T.ToPILImage()(img))