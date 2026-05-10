# =========================================== LIBRARIES ===========================================
import argparse
import importlib

from models import buildModel

from tools import getDevice, InferenceCLI

from pathlib import Path
from torchvision.utils import save_image
from PIL import Image
import torch

from rich.traceback import install             # For a better visualization of the errors
install()
# =================================================================================================

args = InferenceCLI().parse()
config = args.config


# ---------------------------- PARAMETERS ----------------------------
checkpointPath = Path(args.checkpoint)
savingDirectory = Path(args.outputDir)
imageToGenerate = args.numImages
imageSize = args.imageSize
# --------------------------------------------------------------------


device = getDevice()

#* Build the model from config (model-agnostic)
model = buildModel(config)

#* Load the Checkpoint
checkpoint = torch.load(checkpointPath, map_location = device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

generatedImages = model.sample(batchSize = imageToGenerate, imageSize = imageSize, device = device)

#* Save the generated Images
for i, image in enumerate(generatedImages):

    savePath = savingDirectory / f"Image_{i}.png"
    save_image(image, savePath, normalize = True, value_range = (-1, 1))
    print(f"The generated image has been correctly saved in '{savePath}'.")

    # Display the image
    img = Image.open(savePath)
    img.show()