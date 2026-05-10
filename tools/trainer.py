# ====================================== LIBRARIES ======================================
import torch
from torch import nn
from torch.optim import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from torch.utils.data import DataLoader

import logging
from rich.text import Text
from rich.panel import Panel
from rich.console import Console
from rich.logging import RichHandler

import time
import datetime
from pathlib import Path
from typing import Optional, Tuple
# =======================================================================================


# --------- LOGGER SETUP ---------
logging.basicConfig(
                        level = "NOTSET",
                        format = "%(message)s",
                        datefmt = '[%d/%m/%Y - %H:%M:%S]',
                        handlers = [RichHandler(rich_tracebacks = True, markup = True, show_level = False)]
                    )
log = logging.getLogger("rich")


class PlainTextFormatter(logging.Formatter):

    '''
        Since the Rich Logger is used to enable markdown logging in the console for a better visualization, this formatter enables the remotion of those tags when the logger must write in teh log file.
    '''

    def format(self, record):

        originalMessage = record.msg
        
        # Strip the markup using Rich's Text class
        if isinstance(record.msg, str):
            record.msg = Text.from_markup(record.msg).plain

        else:

            try:
                
                console = Console(width = 120, color_system = None)
                
                with console.capture() as capture:
                    console.print(record.msg)

                record.msg = capture.get().strip()

            except Exception:
                pass
            
        # Format the record as usual
        result = super().format(record)
        
        # Restore the original message so RichHandler can still use it for the console
        record.msg = originalMessage

        return result


class DiffusionTrainer:

    def __init__(self, model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, lossFunction: nn.Module, device: torch.device, ampType: Optional[torch.dtype], gradientAccumulationSteps: int = 1, checkpointSavingFrequency: int = 5, maxNumCheckpoints: int = 1, checkpointPathRestart: Optional[Path] = None, scheduler: Optional[LRScheduler] = None, maxGradNorm: Optional[float] = 10.0, logPath: Optional[Path] = None, checkpointDirectory: Optional[Path] = None):

        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lossFunction = lossFunction
        self.device = device
        
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.checkpointSavingFrequency = checkpointSavingFrequency
        self.maxNumCheckpoints = maxNumCheckpoints
        self.checkpointPathRestart = checkpointPathRestart
        self.amp = ampType
        self.scaler = self.scaler = GradScaler() if ampType == torch.float16 else None
        self.scheduler = scheduler
        self.maxGradNorm = maxGradNorm
        
        # Setup directories and logging
        self.logPath, self.trainingFolder = self._defineSavingPaths(logPath)
        self.checkpointDirectory = checkpointDirectory or self.trainingFolder
        self.fileHandler = self._attachFileLogger()

        self.startEpoch = 1

        self._resumeFromCheckpoint()


    def _defineSavingPaths(self, logPath: Optional[Path]) -> Tuple[Path, Path]:

        if logPath is not None:
            logPath.parent.mkdir(parents = True, exist_ok = True)
            return logPath, logPath.parent
        
        BASE_DIR = Path(".").resolve()
        trainingFolder = BASE_DIR / "logs" / self.model._get_name() / time.strftime('%d-%m-%Y_(%H-%M-%S)')
        trainingFolder.mkdir(parents = True, exist_ok = True)

        return trainingFolder / "training.log", trainingFolder


    def _attachFileLogger(self) -> logging.FileHandler:

        fileHandler = logging.FileHandler(self.logPath)
        fileFormatter = PlainTextFormatter('[%(asctime)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S')
        fileHandler.setFormatter(fileFormatter)
        log.addHandler(fileHandler)

        return fileHandler


    def _saveCheckpoint(self, epoch: int, epochLoss: float):

        checkpointPath = self.checkpointDirectory / f"Epoch_{epoch}.pth"

        checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "loss": epochLoss
                    }
        
        torch.save(checkpoint, checkpointPath)

        checkpoints = sorted(self.checkpointDirectory.glob("Epoch_*.pth"), key = lambda p: int(p.stem.split("_")[1]))

        if len(checkpoints) > self.maxNumCheckpoints:
            checkpoints[0].unlink()

        log.info(f"[bold blue][INFO][/bold blue] The checkpoint 'Epoch_{epoch}.pth' has been correctly saved in: {checkpointPath}.\n")

    
    def _resumeFromCheckpoint(self):

        if self.checkpointPathRestart is None: 
            return

        if not self.checkpointPathRestart.exists():
            log.warning(f"[bold red][WARNING][/bold red] Checkpoint {self.checkpointPathRestart} not found. Starting Training from scratch.")
            return

        checkpoint = torch.load(self.checkpointPathRestart, map_location = self.device)

        log.info(f"[bold blue][INFO][/bold blue] Resuming training from checkpoint: {self.checkpointPathRestart}")
        log.info(f'[bold blue][INFO][/bold blue] Epoch: {checkpoint["epoch"]}   Loss: {checkpoint["loss"]:.6f}')

        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        self.startEpoch = checkpoint["epoch"] + 1


    def _printTrainConfiguration(self, epochs: int):

        '''
        Logs the Model Architecture and the Training Parameters used at configuration time before starting the training loop.
        '''

        # Use a plain console to render the panels into strings that can be used by the logger
        dummyConsole = Console(width = 100, color_system = None)

        # Model Architecture Box
        model = str(self.model)
        modelBox = Panel(model, title = "MODEL ARCHITECTURE")
        
        with dummyConsole.capture() as capture:
            dummyConsole.print(modelBox)

        modelBoxSTR = capture.get().strip()

        # Training Parameters Box
        trainingParam = (

                            f"{'Training Epochs:':<30} {self.startEpoch} -> {epochs}\n"
                            f"{'Batch Size:':<30} {self.dataloader.batch_size if hasattr(self.dataloader, 'batch_size') else 'N/A'}\n"
                            f"{'Optimizer:':<30} {self.optimizer.__class__.__name__}\n"
                            f"{'Learning Rate:':<30} {self.optimizer.param_groups[0]['lr']} (Initial)\n"
                            f"{'Scheduler:':<30} {self.scheduler.__class__.__name__ if self.scheduler else 'None'}\n"
                            f"{'Loss Function:':<30} {self.lossFunction.__class__.__name__}\n"
                            f"{'Grad Accumulation Steps:':<30} {self.gradientAccumulationSteps}\n"
                            f"{'Max Grad Norm:':<30} {self.maxGradNorm}\n"
                            f"{'AMP (Mixed Precision):':<30} {f'Enabled ({self.amp})' if self.amp else 'Disabled'}\n"
                            f"{'Checkpoint Frequency:':<30} Every {self.checkpointSavingFrequency} epochs\n"
                            f"{'Device:':<30} {self.device}"

                        )

        trainingBox = Panel(trainingParam, title = "TRAINING")
        
        with dummyConsole.capture() as capture:
            dummyConsole.print(trainingBox)
        trainingBoxSTR = capture.get().strip()

        log.info(f"\n[bold cyan]{modelBoxSTR}[/bold cyan]\n")
        log.info(f"\n[bold green]{trainingBoxSTR}[/bold green]\n\n")


    def train(self, epochs: int):

        self._printTrainConfiguration(epochs)
        
        log.info(f"[bold blue][INFO][/bold blue] Starting Training for '{self.model._get_name()}' Model. Logs will be stored in '{self.logPath}'.\n")
        numBatches = len(self.dataloader)

        try:

            for epoch in range(self.startEpoch, epochs + 1):

                self.model.train()
                self.optimizer.zero_grad()

                epochLoss = 0
                epochStart = time.time()

                for i, batch in enumerate(self.dataloader):

                    batch = batch.to(self.device)

                    # Forward pass
                    with torch.autocast(device_type = self.device.type, dtype = self.amp, enabled = self.amp is not None):

                        target, prediction = self.model(batch)
                        loss = self.lossFunction(prediction, target)
                        loss = loss / self.gradientAccumulationSteps

                    epochLoss += loss.item() * self.gradientAccumulationSteps

                    # Backward pass
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        
                    else:
                        loss.backward()

                    # Optimization step
                    gradNorm = 'N/A'
                    if (i + 1) % self.gradientAccumulationSteps == 0 or (i + 1) == numBatches:

                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)

                        clipValue = self.maxGradNorm if self.maxGradNorm is not None else float('inf')
                        gradNorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipValue)
                        gradNorm = f"{gradNorm.item():.4f}"
                        
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                        else:
                            self.optimizer.step()
                        
                        self.optimizer.zero_grad()

                        if self.scheduler is not None:
                            self.scheduler.step()

                    elapsed = time.time() - epochStart
                    timePerBatch = elapsed / (i + 1)

                    # Epoch ETA
                    etaEpoch = timePerBatch * (numBatches - (i + 1))
                    etaEpoch = str(datetime.timedelta(seconds = int(etaEpoch)))

                    # Global ETA
                    totalRemainingBatches = (numBatches - (i + 1)) + ((epochs - epoch) * numBatches)
                    etaGlobal = timePerBatch * totalRemainingBatches
                    etaGlobal = str(datetime.timedelta(seconds = int(etaGlobal)))

                    log.info(f"[bold green][TRAIN][/bold green] Epoch: {epoch}/{epochs} | Batch: {i + 1}/{numBatches} | ETA: ({etaEpoch}) [{etaGlobal}] | LR: {self.optimizer.param_groups[0]['lr']:.6f} | Loss: {epochLoss/(i + 1):.6f} | GradNorm: {gradNorm}")

                # Save Checkpoint
                if (epoch == 1) or (epoch % self.checkpointSavingFrequency == 0):
                    self._saveCheckpoint(epoch, epochLoss / numBatches)

                log.info(f"[bold green][TRAIN][/bold green] Epoch: {epoch}/{epochs} finished!\n")

        except Exception as e:
            log.exception(f"[bold red][ERROR][/bold red] An error occurred during training: {str(e)}")

        finally:

            log.removeHandler(self.fileHandler)
            self.fileHandler.close()
