# ============================== Dataset Registry ==============================
# Maps string names to dataset classes. Each dataset module registers itself
# when imported.
#
# Usage:
#           from datasets import getDataset
#           DatasetClass = getDataset("celebahq")
#           dataset = DatasetClass(...)
# ==============================================================================

DATASET_REGISTRY = {}


def registerDataset(name: str):

    '''
        Decorator that registers a dataset class in the global registry.

        Parameters
        ----------
        name : str
            The name to register the dataset under (e.g. "celebahq").
    '''

    def decorator(cls):

        DATASET_REGISTRY[name] = cls
        return cls
    
    return decorator


def getDataset(name: str):

    '''
        Retrieve a registered dataset class by name.

        Parameters
        ----------
        name : str
            The registered name of the dataset.

        Returns
        -------
        dataset
            The dataset class.

        Raises
        ------
        KeyError
            If the dataset name is not found in the registry.
    '''

    if name not in DATASET_REGISTRY:

        raise KeyError(f"Dataset '{name}' not found in the registry. The available datasets are: {list(DATASET_REGISTRY.keys())}")
    
    return DATASET_REGISTRY[name]


# Import the dataset modules to trigger registration
import datasets.celebahq