import pytest
import torch

from cgback.schedulers import BaseScheduler, LinearScheduler, CosineScheduler


@pytest.fixture
def linear_scheduler():
    """
    Fixture for initializing a LinearScheduler for tests.
    """
    return LinearScheduler(num_timesteps=100, beta_start=0.1, beta_end=0.2)


def test_base_scheduler_abstract_instantiation():
    """
    Ensure that the BaseScheduler cannot be instantiated directly.
    """
    with pytest.raises(TypeError) as excinfo:
        BaseScheduler(1000, 0.01, 0.02)

    assert "Can't instantiate abstract class" in str(excinfo.value)


def test_base_scheduler_abstract_method():
    """
    Ensure the abstract method calculate_parameters raises an error if called.
    """

    class DummyScheduler(BaseScheduler):
        def __init__(self):
            super().__init__(1000, 0.01, 0.02)

    with pytest.raises(TypeError) as excinfo:
        DummyScheduler()

    assert "instantiate abstract class" in str(excinfo.value)


def test_linear_scheduler_initialization():
    """
    Test that LinearScheduler initializes correctly and computes betas, alphas, and alpha_bars.
    """
    num_timesteps = 100
    beta_start = 0.001
    beta_end = 0.02

    scheduler = LinearScheduler(num_timesteps, beta_start, beta_end)

    # Ensure betas, alphas, alpha_bars are tensors of the correct shape
    assert scheduler.betas.shape == (num_timesteps,)
    assert scheduler.alphas.shape == (num_timesteps,)
    assert scheduler.alpha_bars.shape == (num_timesteps,)

    # Check the range of betas
    assert torch.all(scheduler.betas >= beta_start)
    assert torch.all(scheduler.betas <= beta_end)


def test_cosine_scheduler_initialization():
    """
    Test that CosineScheduler initializes correctly and computes betas, alphas, and alpha_bars.
    """
    num_timesteps = 100
    max_beta = 0.99
    scheduler = CosineScheduler(num_timesteps, max_beta)

    # Ensure betas, alphas, alpha_bars are tensors of the correct shape
    assert scheduler.betas.shape == (num_timesteps,)
    assert scheduler.alphas.shape == (num_timesteps,)
    assert scheduler.alpha_bars.shape == (num_timesteps,)

    # Ensure betas do not exceed max_beta
    assert torch.all(scheduler.betas <= max_beta)


def test_linear_scheduler_extreme_values():
    """
    Test LinearScheduler with extreme values.
    """
    scheduler = LinearScheduler(num_timesteps=1, beta_start=0.1, beta_end=0.1)

    # Check that betas are computed correctly even with single timestep
    assert scheduler.betas.shape == (1,)
    assert scheduler.betas[0] == 0.1


def test_cosine_scheduler_extreme_values():
    """
    Test CosineScheduler with edge cases for small timesteps and large values.
    """
    scheduler = CosineScheduler(num_timesteps=1, max_beta=0.999)

    assert scheduler.betas.shape == (1,)
    assert scheduler.betas[0] <= 0.999


def test_add_noise(linear_scheduler):
    """
    Test the `add_noise` function to ensure it correctly adds noise to the input tensor.
    """
    scheduler = linear_scheduler

    # Create input tensors
    x = torch.ones((5, 5), dtype=torch.float32)  # Input tensor
    noise = torch.randn_like(x)  # Random noise tensor
    t = torch.tensor([10])  # A specific timestep as a tensor

    # Add noise using the scheduler
    x_noisy = scheduler.add_noise(x, t, noise)

    # Compute expected noisy tensor
    alpha_bar_t = scheduler.alpha_bars[t.item() - 1]
    expected = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

    # Assertions
    assert x_noisy.shape == x.shape, "Output shape must match input shape."
    assert torch.allclose(x_noisy, expected), "Noised tensor does not match expected values."


def test_add_remove_noise_with_node_wise_timesteps(linear_scheduler):
    """
    Test the `add_noise` and `remove_noise` functions with node-wise timesteps.
    """
    scheduler = linear_scheduler
    n_nodes = 10

    # Create input tensors
    x = torch.rand((n_nodes, 3), dtype=torch.float32)
    t = torch.randint(1, scheduler._num_timesteps + 1, (n_nodes,))
    noise_inference = torch.randn((n_nodes, 3), dtype=torch.float32)
    noise_sampling = torch.randn((n_nodes, 3), dtype=torch.float32)

    # Test addition of noise in batch
    x_noisy = scheduler.add_noise(x, t, noise_inference)
    x_noisy_expected = torch.cat([
        scheduler.add_noise(x[i:i + 1], t[i:i + 1], noise_inference[i:i + 1])
        for i in range(n_nodes)
    ], dim=0)
    assert torch.allclose(x_noisy, x_noisy_expected)

    # Test removal of noise in batch
    x_denoised = scheduler.remove_noise(x, t, noise_inference, noise_sampling)
    x_denoised_expected = torch.cat([
        scheduler.remove_noise(x[i:i + 1], t[i:i + 1], noise_inference[i:i + 1], noise_sampling[i:i + 1])
        for i in range(n_nodes)
    ], dim=0)
    assert torch.allclose(x_denoised, x_denoised_expected)


def test_add_noise_edge_cases(linear_scheduler):
    """
    Test the `add_noise` function for edge cases like the first and last timestep.
    """
    scheduler = linear_scheduler
    x = torch.ones((4, 4), dtype=torch.float32)
    noise = torch.randn_like(x)

    # Test first timestep
    t = torch.tensor([1])
    x_noisy_first = scheduler.add_noise(x, t, noise)
    alpha_bar_t_first = scheduler.alpha_bars[t.item() - 1]
    assert x_noisy_first.shape == x.shape
    assert torch.allclose(x_noisy_first, torch.sqrt(alpha_bar_t_first) * x + torch.sqrt(1 - alpha_bar_t_first) * noise)

    # Test last timestep
    t = torch.tensor([scheduler._num_timesteps])
    x_noisy_last = scheduler.add_noise(x, t, noise)
    alpha_bar_t_last = scheduler.alpha_bars[t.item() - 1]
    assert x_noisy_last.shape == x.shape
    assert torch.allclose(x_noisy_last, torch.sqrt(alpha_bar_t_last) * x + torch.sqrt(1 - alpha_bar_t_last) * noise)


def test_add_noise_invalid_timestep(linear_scheduler):
    """
    Verify that `add_noise` raises an error if given an invalid timestep.
    """
    scheduler = linear_scheduler
    x = torch.ones((4, 4), dtype=torch.float32)
    noise = torch.randn_like(x)

    with pytest.raises(AssertionError):
        scheduler.add_noise(x, torch.tensor([0]), noise)  # Invalid timestep

    with pytest.raises(AssertionError):
        scheduler.add_noise(x, torch.tensor([scheduler._num_timesteps + 1]), noise)  # Invalid timestep


def test_remove_noise_edge_cases(linear_scheduler):
    """
    Test the `remove_noise` function for edge cases like the first timestep.
    """
    scheduler = linear_scheduler
    x = torch.ones((3, 3), dtype=torch.float32)
    noise_inference = torch.randn_like(x)
    noise_sampling = torch.randn_like(x)

    # Test first timestep
    t = torch.tensor([1])
    x_noisy = scheduler.add_noise(x, t, noise_inference)
    x_denoised = scheduler.remove_noise(x_noisy, t, noise_inference, noise_sampling)

    # At t=1, the noise_sampling should ideally result in no randomness
    assert torch.allclose(x_denoised, x, atol=1e-5), "Denoised tensor at t=1 should approximately equal the original input."


def test_remove_noise_invalid_timestep(linear_scheduler):
    """
    Verify that `remove_noise` raises an error if given an invalid timestep.
    """
    scheduler = linear_scheduler
    x = torch.ones((4, 4), dtype=torch.float32)
    noise_inference = torch.randn_like(x)
    noise_sampling = torch.randn_like(x)

    with pytest.raises(AssertionError):
        scheduler.remove_noise(x, torch.tensor([0]), noise_inference, noise_sampling)  # Invalid timestep

    with pytest.raises(AssertionError):
        scheduler.remove_noise(x, torch.tensor([scheduler._num_timesteps + 1]), noise_inference, noise_sampling)  # Invalid timestep
