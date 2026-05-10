# ================================== Model Registry ==================================
# Maps the string names to the model classes and their builder functions. 
# Each model package registers itself when imported (e.g. models/ddpm/__init__.py).
#
# Usage:
#           from models import getModel, buildModel
#           DDPMClass = getModel("ddpm")             # Get the class
#           model = buildModel(config)               # Build a full model from config
# ====================================================================================

from typing import Callable, Optional

MODEL_REGISTRY = {}


def registerModel(name: str, builder: Optional[Callable] = None):

    '''
        Decorator that registers a model class (and optionally its builder) in the global registry.

        Parameters
        ----------
        name : str
            The name to register the model under (e.g. "ddpm").

        builder : callable, optional
            A factory function with signature `builder(config) -> nn.Module` that knows how to fully construct the model from a Configuration Object.
            When provided, the model can be built generically via `buildModel(config)`.

        Example
        -------
         @registerModel("ddpm", builder = buildDDPM)
         class DDPM(BaseDiffusion):
            ...
    '''

    def decorator(cls):

        MODEL_REGISTRY[name] = {"class": cls, "builder": builder}

        return cls
    
    return decorator


def getModel(name: str):

    '''
        Retrieve a registered model class by name.

        Parameters
        ----------
        name : str
            The registered name of the model.

        Returns
        -------
        model
            The model class.

        Raises
        ------
        KeyError
            If the model name is not found in the registry.
    '''

    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found in the registry. The available models are: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[name]["class"]


def buildModel(config):

    '''
        Build a complete model instance from a Configuration Object using the registered builder.
        The builder is looked up by `config.model.name`.

        Parameters
        ----------
        config : Config
            The top-level configuration dataclass (must have `config.model.name`).

        Returns
        -------
        nn.Module
            The fully constructed model, ready for training or inference.

        Raises
        ------
        KeyError
            If the model name is not found in the registry.

        ValueError
            If the model has no registered builder function.
    '''

    name = config.model.name

    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found in the registry. The available models are: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[name]

    if entry["builder"] is None:
        raise ValueError(
            f"Model '{name}' has no registered builder. "
            f"[ModelPackage/__init__.py] Register one with: registerModel('{name}', builder = myBuilderFunction)(ModelClass)"
        )
    
    return entry["builder"](config)