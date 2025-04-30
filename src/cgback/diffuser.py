import sys
import enum
import torch
from torch import nn, Tensor
from torch.nn import Linear
from rich.console import Console
from rich.progress import Progress
from abc import ABC, abstractmethod
from cgback.egnn import EGNN, CompleteLocalFrameEGNN
from cgback.dataloader import GraphDataloaderItem
from cgback.schedulers import LinearScheduler, CosineScheduler


class Scheduler(enum.Enum):
    LINEAR = "linear"
    COSINE = "cosine"


class BaseDiffuser(nn.Module):
    def __init__(self, num_timesteps: int, dim_features: int, dim_embedding: int, num_layers: int, scheduler_type: str) -> None:
        super().__init__()
        self._num_timesteps = num_timesteps
        self.dim_features = dim_features
        self.dim_embedding = dim_embedding
        self.num_layers = num_layers
        self.scheduler_type = scheduler_type

        match Scheduler(scheduler_type):
            case Scheduler.LINEAR:
                self.scheduler = LinearScheduler(num_timesteps=num_timesteps)
            case Scheduler.COSINE:
                self.scheduler = CosineScheduler(num_timesteps=num_timesteps)

        self.features_embedding = Linear(dim_features + 1, dim_embedding)
        # self.egnn = EGNN(dim_features=dim_embedding, num_layers=num_layers)
        self.egnn = CompleteLocalFrameEGNN(dim_features=dim_embedding, num_layers=num_layers)

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, num_timesteps: int):
        self._num_timesteps = num_timesteps
        match Scheduler(self.scheduler_type):
            case Scheduler.LINEAR:
                self.scheduler = LinearScheduler(num_timesteps=self.num_timesteps)
            case Scheduler.COSINE:
                self.scheduler = CosineScheduler(num_timesteps=self.num_timesteps)

    @abstractmethod  # Must be implemented by subclasses
    def get_noise(self, xw: Tensor) -> Tensor:
        pass

    def forward(self, data: GraphDataloaderItem, device: str | torch.device = "cpu") -> tuple[Tensor, int]:
        # Unpack data
        hu = data.hu.to(device)
        xw = data.xw.to(device)
        e = data.e.to(device)
        num_h = data.num_h
        mask = data.mask.to(device)

        # Calculate noise
        noise = torch.randn_like(xw).to(device)

        # Sample a time step
        t = torch.randint(1, self.num_timesteps + 1, (1,), device=device)

        # Add noise
        xw_noisy = self.scheduler.add_noise(xw, t, noise, mask)

        # Encode time information
        t_encoded = (t / self.num_timesteps) * torch.ones((hu.shape[0], 1), device=device)
        hut = torch.cat((hu, t_encoded), dim=-1)
        hut_embedding = self.features_embedding(hut)

        # Predict noise
        _, noise_predicted = self.egnn(hut_embedding, xw_noisy, e)
        noise_predicted = noise_predicted

        # Mask noise
        noise = noise * mask
        noise_predicted = noise_predicted * mask

        # Calculate loss
        loss_sum = nn.functional.mse_loss(noise, noise_predicted, reduction="sum")
        loss_cnt = torch.sum(num_h).item() * 3

        return loss_sum, loss_cnt

    @torch.inference_mode()
    def sample(self, data: GraphDataloaderItem, device: str | torch.device = "cpu", verbose: bool = True, bar_description: str = None) -> GraphDataloaderItem:
        with torch.no_grad():
            # Unpack data
            hu = data.hu.to(device)
            xw = data.xw.to(device)
            e = data.e.to(device)
            mask = data.mask.to(device)

            # Initialize the time step
            t = torch.zeros(1, device=device, dtype=torch.int64)

            # Denoise positions
            with Progress(console=Console(file=sys.stderr), disable=not verbose) as progress:
                if verbose: task = progress.add_task(bar_description if bar_description else "[cyan]Denoising graphs...", total=self.num_timesteps)
                for i in range(self.num_timesteps, 0, -1):
                    # Calculate noise
                    noise = self.get_noise(xw)

                    # Set time step
                    t.fill_(i)

                    # Encode time information
                    t_encoded = (t / self.num_timesteps) * torch.ones((hu.shape[0], 1)).to(device)
                    hut = torch.cat((hu, t_encoded), dim=-1)
                    hut_embedding = self.features_embedding(hut)

                    # Predict noise
                    _, noise_inference = self.egnn(hut_embedding, xw, e)

                    # Remove noise
                    xw = self.scheduler.remove_noise(xw, t, noise_inference, noise, mask)

                    if verbose: progress.update(task, advance=1)

        # Build new graph_dict
        new_data = data
        new_data.xw = xw

        return new_data

class DDPM(BaseDiffuser):
    def __init__(self, num_timesteps: int, dim_features: int, dim_embedding: int, num_layers: int, scheduler_type: str) -> None:
        super().__init__(num_timesteps, dim_features, dim_embedding, num_layers, scheduler_type)

    def get_noise(self, xw: Tensor) -> Tensor:
        return torch.randn_like(xw).to(xw.device)

class DDIM(BaseDiffuser):
    def __init__(self, num_timesteps: int, dim_features: int, dim_embedding: int, num_layers: int, scheduler_type: str) -> None:
        super().__init__(num_timesteps, dim_features, dim_embedding, num_layers, scheduler_type)

    def get_noise(self, xw: Tensor) -> Tensor:
        return torch.zeros_like(xw).to(xw.device)