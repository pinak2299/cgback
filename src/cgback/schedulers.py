import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module


class BaseScheduler(ABC, Module):
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        self._num_timesteps = num_timesteps
        self._beta_start = beta_start
        self._beta_end = beta_end

    @abstractmethod  # Must be implemented by subclasses
    def calculate_parameters(self) -> None:
        pass

    @property
    def betas(self) -> Tensor:
        return self._betas

    @property
    def alphas(self) -> Tensor:
        return self._alphas

    @property
    def alpha_bars(self) -> Tensor:
        return self._alpha_bars

    def add_noise(self, x: Tensor, t: Tensor, noise: Tensor, mask: Tensor | None = None) -> Tensor:
        assert torch.all((1 <= t) & (t <= self._num_timesteps))

        t_idx = t - 1
        alpha_bar = self._alpha_bars[t_idx].view(-1, 1)
        if mask is not None:
            x_noisy = (1 - mask) * x + mask * (torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise)
        else:
            x_noisy = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise

        return x_noisy

    def remove_noise(self, x: Tensor, t: Tensor, noise_inference: Tensor, noise_sampling: Tensor, mask: Tensor | None = None) -> Tensor:
        assert torch.all((1 <= t) & (t <= self._num_timesteps))

        t_idx = t - 1
        alpha = self._alphas[t_idx].view(-1, 1)
        alpha_bar = self._alpha_bars[t_idx].view(-1, 1)
        alpha_bar_prev = self._alpha_bars[t_idx - 1].view(-1, 1)

        zero_mask = (t == 1).expand(noise_sampling.shape[0])
        noise_sampling[zero_mask] = 0

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_inference) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        if mask is not None:
            x_denoised = (1 - mask) * x + mask * (mu + noise_sampling * std)
        else:
            x_denoised = mu + noise_sampling * std

        return x_denoised


class LinearScheduler(BaseScheduler):
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02) -> None:
        super().__init__(num_timesteps, beta_start, beta_end)
        self.calculate_parameters()

    def calculate_parameters(self) -> None:
        betas = torch.linspace(self._beta_start, self._beta_end, self._num_timesteps)
        alphas = 1.0 - betas  # Computes corresponding alpha values
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("_betas", betas)
        self.register_buffer("_alphas", alphas)
        self.register_buffer("_alpha_bars", alpha_bars)


class CosineScheduler(BaseScheduler):
    def __init__(self, num_timesteps: int = 1000, max_beta: float = 0.999, offset: float = 8e-3, exponent: int = 2) -> None:
        super().__init__(num_timesteps, 0, max_beta)
        self._offset = offset
        self._exponent = exponent
        self.calculate_parameters()

    def calculate_parameters(self) -> None:
        def alpha_bar_fn(t):
            # Implements the cosine-based alpha_bar function
            return math.cos((t + self._offset) / (1.0 + self._offset) * math.pi / 2) ** self._exponent

        betas = []
        max_beta = self._beta_end
        for i in range(self._num_timesteps):
            t1 = i / self._num_timesteps
            t2 = (i + 1) / self._num_timesteps
            beta = min(1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta)
            betas.append(beta)

        betas = torch.tensor(betas, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("_betas", betas)
        self.register_buffer("_alphas", alphas)
        self.register_buffer("_alpha_bars", alpha_bars)
