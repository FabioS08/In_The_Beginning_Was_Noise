# ============================== Model Registry ==============================
# Maps string names to model classes. Each model package registers itself
# when imported (e.g., models/ddpm/__init__.py).
#
# Usage:
#           from models import getModel
#           DDPMClass = getModel("ddpm")
#           model = DDPMClass(...)
# ============================================================================

MODEL_REGISTRY = {}


def registerModel(name: str):

    '''
        Decorator that registers a model class in the global registry.

        Parameters
        ----------
        name : str
            The name to register the model under (e.g. "ddpm").

        Example
        -------
         @register_model("ddpm") 

         class DDPM(BaseDiffusion):
            ...
    '''

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
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
    
    return MODEL_REGISTRY[name]


# Import model packages to trigger registration
import models.ddpm
