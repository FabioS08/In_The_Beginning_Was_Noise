# ====================================== LIBRARIES ======================================
import sys
import argparse
import importlib
from abc import ABC, abstractmethod

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align

console = Console()
# =======================================================================================


class BaseCLI(ABC):

    '''
        Base class for CLI tools in this package. It provides a structured way to define and parse command-line arguments as well as a consistent help message format using the rich library.

        Arguments
        ---------
        console: Optional[Console]
            An optional Console object to be used for printing. If not provided, a new Rich Console will be created.
    '''

    def __init__(self, console = None):
        self.console = console or Console()
        self.parser = argparse.ArgumentParser(add_help = False)

        self.title = "CLI Tool"
        self.description = "Command line interface."
        self.note = ""
        self.usageExample = "python script.py [OPTIONS]"

        self._setupArguments()


    @abstractmethod
    def _setupArguments(self):

        '''
            Define the custom arguments that must be parsed.
        '''
        ...


    @abstractmethod
    def _getHelpRows(self):
        
        '''
            Define the help table rows, where each row must follow the format: (argument, description)
        '''

        return []


    def _printHelp(self):

        '''
            Print the help message using the rich library for a better formatting.
        '''

        w = self.console.size.width - 40

        body = f"{self.description}\n"

        if self.note:
            body += f"[dim]{self.note}[/]\n\n"

        body += f"Usage Example: [blue]{self.usageExample}[/blue]"

        self.console.print(
                                Align.center(

                                    Panel(
                                                body,
                                                title = f"[bold cyan]🚀 {self.title}[/]\n",
                                                border_style = "cyan",
                                                width = w,
                                            )
                                )
                            )

        table = Table(title = "\nArguments", show_lines = True, width = w)
        table.add_column("Argument", style = "bold green")
        table.add_column("Description", style = "white")

        for arg, desc in self._getHelpRows():
            table.add_row(arg, desc)

        self.console.print(Align.center(table))


    def _loadConfig(self, moduleName):

        '''
            Dynamically import the configuration module specified by the user and return the 'config' variable contained in it.
        '''

        try:

            module = importlib.import_module(moduleName)
        
        except ModuleNotFoundError:

            self.console.print(f"[bold red]Error:[/] The Configuration Module '{moduleName}' has not been found!")
            sys.exit(1)

        if not hasattr(module, "config"):
           
            self.console.print(f"[bold red]Error:[/] The Module '{moduleName}' has no [bold]config[/] variable!")
            sys.exit(1)

        return module.config


    @abstractmethod
    def parse(self):

        '''
            Parse the CLI Arguments
        '''

        args = self.parser.parse_args()

        if getattr(args, "help", False):

            self._printHelp()
            sys.exit(0)

        return args


class TrainCLI(BaseCLI):

    '''
        CLI class for the training script.
    '''

    def __init__(self, console = None):

        super().__init__(console)

        self.title = "Diffusion Model Trainer"
        self.description = "Train a diffusion model using a configuration module."
        self.note = "Note that in this package, the modules are placed in the 'configs' directory."
        self.usageExample = "python train.py --config configs.ddpm_celebahq [--printConfig]"


    def _setupArguments(self):

        self.parser.add_argument("--config", type = str)
        self.parser.add_argument("--printConfig", action = "store_true")
        self.parser.add_argument("-h", "--help", action = "store_true")


    def _getHelpRows(self):

        return [
                    (
                        "--config MODULE",
                        "Path to the selected configuration module.\n"
                        "[dim]Note that the module must contain a variable called 'config'.[/]"
                    ),

                    (
                        "--printConfig",
                        "Print the loaded configuration.\n"
                        "[dim]The path to the configuration module is obviously required to use this option.[/]"
                    ),

                    (
                        "-h, --help",
                        "Show the help message."
                    ),
                ]


    def parse(self):

        args = self.parser.parse_args()

        if args.help or args.config is None:

            self._printHelp()
            sys.exit(0)

        config = self._loadConfig(args.config)

        if args.printConfig:

            self.console.print(
                                    Align.center(
                                                    Panel(
                                                            Align.center("[bold cyan] >>> The loaded Configuration Module is shown below <<< [/]"),
                                                            border_style="cyan",
                                                            width=self.console.size.width - 40,
                                                    )
                                                )
                                )
            self.console.print("\n")
            self.console.print(config)
            sys.exit(0)

        return config


class InferenceCLI(BaseCLI):

    '''
        CLI class for the inference script.
    '''

    def __init__(self, console = None):

        super().__init__(console)

        self.title = "Diffusion Model Inference"
        self.description = "Run inference with a trained diffusion model."
        self.note = "Note that in this package, the modules are placed in the 'configs' directory."
        self.usageExample = "python inference.py --config configs.ddpm_celebahq --checkpoint checkpoints/ddpm_celebahq.pth [--printConfig] [--outputDir path/to/outputDirectory] [--numImages N] [--imageSize X | W,H]"


    def imageSizeinput(self, value):

        '''
            Custom argument type to parse the image size, which can be either a single integer (e.g. 256) or a tuple of two integers (e.g. '256,256').
        '''

        # Check if it's a comma-separated string
        if ',' in value:

            try:
                return tuple(int(x.strip()) for x in value.split(','))
            
            except ValueError:
                raise argparse.ArgumentTypeError("Tuple must be integers separated by commas (e.g. '256,256')")
        
        try:
        
            return int(value)
        
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer or integer-tuple")


    def _setupArguments(self):

        self.parser.add_argument("--config", type = str)
        self.parser.add_argument("--checkpoint", type = str)
        self.parser.add_argument("--printConfig", action = "store_true")
        self.parser.add_argument("--outputDir", type = str, default = ".")
        self.parser.add_argument("--numImages", type = int, default = 1)
        self.parser.add_argument("--imageSize", type = self.imageSizeinput, default = 256)
        self.parser.add_argument("-h", "--help", action = "store_true")


    def _getHelpRows(self):

        return [
                    (
                        "--config MODULE",
                        "Path to the selected configuration module.\n"
                        "[dim]Note that the module must contain a variable called 'config'.[/]"
                    ),

                    (
                        "--checkpoint path/to/checkpoint.pth",
                        "Load the specified model checkpoint.\n"
    
                    ),

                    (
                        "--printConfig",
                        "Print the loaded configuration.\n"
                        "[dim]The path to the configuration module is obviously required to use this option.[/]"
                    ),

                    (

                        "--outputDir path/to/outputDirectory",
                        "Specify the output directory where the generated images will be saved. If not provided, the current directory will be used."

                    ),

                    (

                        "--numImages N",
                        "Specify the number of images to generate. If not provided, just one image will be generated."

                    ),

                    (

                        "--imageSize Size",
                        "Specify the size of the generated images (either as a single integer element or as a tuple if you want to specify different width and height by using a comma-separated value). If not provided, the images will use a resolution of 256x256."

                    ),

                    (
                        "-h, --help",
                        "Show the help message."
                    ),
                ]


    def parse(self):

        args = self.parser.parse_args()

        if args.help or args.config is None:

            self._printHelp()
            sys.exit(0)

        config = self._loadConfig(args.config)

        if args.printConfig:


            self.console.print(
                                    Align.center(
                                                    Panel(
                                                                Align.center("[bold cyan] >>> The loaded Configuration Module is shown below <<< [/]"),
                                                                border_style = "cyan",
                                                                width = self.console.size.width - 40,
                                                            )
                                                )
                                )
            self.console.print("\n")
            self.console.print(config)
            sys.exit(0)

        return args