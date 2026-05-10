from models import registerModel
from models.ddpm.ddpm import DDPM
from models.ddpm.config import encoderConfig, bottleneckConfig, decoderConfig, initChannels
from models.networks.embeddings import TimeStepEmbedding
from models.networks.unet import UNet
from tools.trainingUtils import selectNoiseScheduler


def buildDDPM(config):

    '''
        Build the DDPM model from a Configuration Object.

        This factory function encapsulates all the DDPM-specific construction logic: noise schedule, time embedding, UNet network and the DDPM wrapper.

        Parameters
        ----------
        config : Config
            The top-level configuration dataclass.

        Returns
        -------
        DDPM
            A fully constructed DDPM model instance.
    '''

    noiseSchedule = selectNoiseScheduler(config.model.varianceSchedule, **config.model.varianceScheduleParams)
    timeEmbedding = TimeStepEmbedding(config.model.timeEmbeddingDimension)
    unetModel = UNet(timeStepEmbedding = timeEmbedding, initChannels = initChannels,
                     encoder = encoderConfig, bottleneck = bottleneckConfig, decoder = decoderConfig)
    
    return DDPM(unetModel = unetModel, noiseSchedule = noiseSchedule, T = config.model.timesteps)


registerModel("ddpm", builder = buildDDPM)(DDPM)
