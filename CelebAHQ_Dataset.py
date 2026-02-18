import rich.spinner
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from zipfile import ZipFile
import pathlib
import gdown
from rich import print
import os


class CelebAHQ(Dataset):

    def __init__(self, rootPath: pathlib.Path, download: bool = False, transforms = None):

        self.rootPath = rootPath
        self.tranforms = transforms
        self.fileID = "1jlQ8umhpJo8lVgC9q4_1q_t_Frv1kZ3f"
        self.URL = f"https://drive.google.com/uc?id={self.fileID}"
        self.zipPath = rootPath / 'CelebAHQ.zip'
        self.destPath = rootPath / "CelebA-HQ"

        self.printDatasetInfo()

        # If the dataset has not been downlaoded yet, do it and extract the files
        if download and not any(rootPath.iterdir()):

            try:
                    # Download the zip file from Google Drive
                    print('[green][*] Downloading the dataset...[/green]')
                    gdown.download(self.URL, str(self.zipPath))
                    print('[green][✓] Download completed![/green]\n')

                    # Extract the zip file
                    print('[green][*] Extracting the dataset...[/green]')
                    with ZipFile(self.zipPath, 'r') as zip:
                        zip.extractall(self.rootPath)
                    print('[green][✓] Extraction completed![/green]\n')
                    
                    # Rename the extracted folder for a better structure
                    (pathlib.Path(self.rootPath) / "images").rename(self.destPath)
                    print(f"[green][✓] Dataset correctly saved in: '{self.destPath}'[/green]")

            finally:

                # Clean up the zip file after extraction
                if os.path.exists(self.zipPath):
                    os.remove(self.zipPath)


    def __len__(self):
        return len(os.listdir(self.destPath))


    def __getitem__(self, idx):

        imgPath = sorted(os.listdir(self.destPath), key = lambda x: int(x.split('.')[0]))[idx]
        img = plt.imread(os.path.join(self.destPath, imgPath))

        return img
    

    def printDatasetInfo(self):

        l = max(len(str(self.rootPath)), len(self.URL), len(str(self.zipPath))) // 2

        print('=' * l + ' [bold blue]DATASET INFO[/bold blue] ' + '=' * l)
        print(f'[bold]Root Path[/bold]: [italic]{self.rootPath}[/italic]')
        print(f'[bold]Zip File[/bold]: [italic]{self.zipPath}[/italic]')
        print(f'[bold]URL[/bold]: [italic]{self.URL}[/italic]')
        print('=' * (2 * l + 14) + '\n')


if __name__ == "__main__":

    datasetPath = pathlib.Path('/Users/fabioschiliro/Desktop/DDPM/datasets')
    dataset = CelebAHQ(datasetPath, download = True)
    index = 200

    labelWidth = 35
    print(f'{"Number of images in the dataset:":<{labelWidth}} {len(dataset)}')
    print(f'{"Image Size:":<{labelWidth}} {dataset[0].shape}')
    print(f'{"Index of visualized image:":<{labelWidth}} {index}')

    plt.imshow(dataset[index])
    plt.axis('off')
    #plt.show()