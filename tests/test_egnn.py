import pytest
import torch

from cgback.egnn import EGNN


@pytest.fixture
def egnn_instance():
    """Fixture to initialize an EGNN instance for testing."""
    return EGNN(dim_features=5, num_layers=3)


def test_egnn_initialization():
    """Test if the EGNN class initializes correctly."""
    EGNN(dim_features=5, num_layers=3)


def test_egnn_forward_basic(egnn_instance):
    """Test if the forward method runs successfully with valid inputs."""
    num_nodes = 4
    num_edges = 3

    x = torch.rand(num_nodes, 3)
    h = torch.rand(num_nodes, 5)
    e = torch.randint(0, num_nodes, (2, num_edges))

    new_h, new_x = egnn_instance(h, x, e)

    # Check output dimensions
    assert new_h.shape == h.shape
    assert new_x.shape == x.shape


def test_egnn_large_graph(egnn_instance):
    """Test that the model can handle larger graphs without crashing."""
    num_nodes = 1000
    num_edges = 3000

    x = torch.rand(num_nodes, 3)
    h = torch.rand(num_nodes, 5)
    e = torch.randint(0, num_nodes, (2, num_edges))

    new_h, new_x = egnn_instance(h, x, e)

    # Verify output dimensions
    assert new_h.shape == h.shape
    assert new_x.shape == x.shape


def test_egnn_with_large_inputs(egnn_instance):
    """Test if the model handles very large input values without numerical instability."""
    num_nodes = 4
    num_edges = 3

    # Large values for coordinates and features
    h = torch.full((num_nodes, 5), 1e6, dtype=torch.float32)
    x = torch.full((num_nodes, 3), 1e6, dtype=torch.float32)
    e = torch.randint(0, num_nodes, (2, num_edges))

    new_h, new_x = egnn_instance(h, x, e)

    # Ensure outputs are finite (not NaN or Inf)
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_x).all()


def test_egnn_with_small_inputs(egnn_instance):
    """Test if the model handles very small input values (close to zero) without numerical instability."""
    num_nodes = 4
    num_edges = 3

    # Small values for coordinates and features
    x = torch.full((num_nodes, 3), 1e-6, dtype=torch.float32)
    h = torch.full((num_nodes, 5), 1e-6, dtype=torch.float32)
    e = torch.randint(0, num_nodes, (2, num_edges))

    new_h, new_x = egnn_instance(h, x, e)

    # Ensure outputs are finite (not NaN or Inf)
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_x).all()


def test_egnn_with_zero_inputs(egnn_instance):
    """Test if the model can handle inputs being all zeros."""
    num_nodes = 4
    num_edges = 3

    # Zero values for coordinates and features
    h = torch.zeros((num_nodes, 5), dtype=torch.float32)
    x = torch.zeros((num_nodes, 3), dtype=torch.float32)
    e = torch.randint(0, num_nodes, (2, num_edges))

    new_h, new_x = egnn_instance(h, x, e)

    # Ensure outputs are finite (not NaN or Inf)
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_x).all()


def test_egnn_inf_check(egnn_instance):
    """Test if the model avoids producing infinities when edge distances are zero."""
    num_nodes = 4
    num_edges = 3

    # Coordinates that result in zero distances for all edges
    h = torch.rand((num_nodes, 5), dtype=torch.float32)
    x = torch.zeros((num_nodes, 3), dtype=torch.float32)
    e = torch.randint(0, num_nodes, (2, num_edges))

    new_h, new_x = egnn_instance(h, x, e)

    # Ensure outputs contain no infinities or NaNs
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_x).all()


def test_egnn_zero_distances_stability(egnn_instance):
    """Test numerical stability when coordinates of connected nodes are identical."""
    num_nodes = 4

    # Identical coordinates (all distances zero)
    h = torch.rand((num_nodes, 5), dtype=torch.float32)
    x = torch.zeros((num_nodes, 3), dtype=torch.float32)
    # Self-loops creating zero-edge distances
    e = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])

    new_h, new_x = egnn_instance(h, x, e)

    # Ensure outputs are finite (not NaN or Inf)
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_x).all()


def test_egnn_translation_equivariance(egnn_instance):
    """Test translation equivariance."""
    num_nodes = 4
    num_edges = 3

    x = torch.rand(num_nodes, 3)
    h = torch.rand(num_nodes, 5)
    e = torch.randint(0, num_nodes, (2, num_edges))
    t = torch.rand(1, 3)

    new_h1, new_x1 = egnn_instance(h, x, e)
    new_x1 = new_x1 + t
    new_h2, new_x2 = egnn_instance(h, x + t, e)

    # Check output dimensions
    assert torch.allclose(new_h1, new_h2)
    assert torch.allclose(new_x1, new_x2)


def test_egnn_rotation_equivariance(egnn_instance):
    """Test rotation equivariance."""
    num_nodes = 4
    num_edges = 3

    x = torch.rand(num_nodes, 3)
    h = torch.rand(num_nodes, 5)
    e = torch.randint(0, num_nodes, (2, num_edges))

    # Generate a random quaternion (q0, q1, q2, q3) with unit norm
    q = torch.randn(4)
    q /= q.norm()

    q0, q1, q2, q3 = q

    # Construct the rotation matrix
    R = torch.tensor([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)],
    ])

    new_h1, new_x1 = egnn_instance(h, x, e)
    new_x1 = torch.matmul(new_x1, R.T)
    new_h2, new_x2 = egnn_instance(h, torch.matmul(x, R.T), e)

    # Check output dimensions
    assert torch.allclose(new_h1, new_h2)
    assert torch.allclose(new_x1, new_x2)
