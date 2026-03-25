from models import registerModel
from models.ddpm.ddpm import DDPM

registerModel("ddpm")(DDPM)
