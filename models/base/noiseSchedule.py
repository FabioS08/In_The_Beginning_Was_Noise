# ====================================== LIBRARIES ======================================
import torch
from abc import ABC, abstractmethod
# =======================================================================================


class NoiseSchedule(ABC):

    '''
        Abstract base class for noise schedules used in diffusion models.

        A noise schedule defines how the noise variance (beta) evolves across the diffusion timesteps.

        Subclasses must implement the `__call__` method, which returns the betas tensor for a given number of timesteps.
    '''

    @abstractmethod
    def __call__(self, T: int) -> torch.Tensor:

        '''
            Compute the noise schedule.

            Parameters
            ----------
            T : int
                The total number of diffusion timesteps.

            Returns
            -------
            torch.Tensor
                A 1D tensor of shape [T] containing the beta values.
        '''
        ...


    def __repr__(self):

        params = ", ".join(f"{k} = {v}" for k, v in self.__dict__.items())
        
        return f"{self.__class__.__name__}({params})"


class LinearSchedule(NoiseSchedule):

    '''
        Linear variance schedule.

        The beta values increase linearly from `betaStart` to `betaEnd` over T timesteps.

        Parameters
        ----------
        betaStart : float
            The starting value of beta. [Default: 1e-4]

        betaEnd : float
            The ending value of beta. [Default: 0.02]
    '''

    def __init__(self, betaStart: float = 1e-4, betaEnd: float = 0.02):

        self.betaStart = betaStart
        self.betaEnd = betaEnd


    def __call__(self, T: int) -> torch.Tensor:

        return torch.linspace(self.betaStart, self.betaEnd, T)