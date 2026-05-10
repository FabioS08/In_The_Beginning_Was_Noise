# ====================================== LIBRARIES ======================================
from typing import List, Tuple
from pathlib import Path

import torch
from torch import nn
# =======================================================================================


def plotModelArchitecture(modelName: str, model: nn.Module, inputSize: List | Tuple | torch.Size, format: str, outputDirectory: str) -> None:

    '''
        Plot the architecture of a PyTorch model and save the visualization to the specified directory.

        Parameters
        ----------
        modelName : str
            Name of the model. Used as the output filename.

        model : nn.Module
            PyTorch model to visualize.

        inputSize : List | Tuple | torch.Size
            Input shape(s) required by the model.

        format : str
            Output file format (e.g., "png", "pdf", "svg").

        outputDirectory : str
            Directory where the visualization will be saved.

        Returns
        -------
        None
            The function saves the visualization to disk and returns nothing.
    '''

    from torchview import draw_graph

    if outputDirectory:
        output = Path(outputDirectory) / Path(f'{modelName}.{format}')

    else: 
        output = modelName

    graph = draw_graph(
                        model,
                        input_size = inputSize,
                        expand_nested = False,
                        depth = 1,
                        graph_name = modelName,
                        roll = True,
                        hide_module_functions = True,
                    
                    )

    graph.visual_graph.render(output, format = format, cleanup = True)
    print(f"The '{modelName}' model architecture has been correctly saved!")


def printModelSummary(model: nn.Module, inputSize: List | Tuple | torch.Size, verbose: int = 1) -> None:

    '''
        Print a summary of a PyTorch model's architecture, including layer types, output shapes and parameter counts.

        Parameters
        ----------
        model : nn.Module
            PyTorch model to summarize.

        inputSize : List | Tuple | torch.Size
            Input shape(s) required by the model.

        verbose : int, optional
            Verbosity level for the summary. [Default: 1]

        Returns
        -------
        None
            The function prints the model summary to the console and returns nothing.
    '''

    from torchinfo import summary

    summary(model, input_size = inputSize, verbose = verbose)
