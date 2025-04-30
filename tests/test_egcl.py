import pytest
import torch

from cgback.egnn import EGCL


@pytest.fixture
def egcl_instance():
    """Fixture to initialize an EGCL instance for testing."""
    return EGCL(dim_h=5)


def test_egcl_initialization():
    """Test if the EGCL class initializes correctly."""
    EGCL(dim_h=5)


def test_egcl_forward_basic(egcl_instance):
    """Test if the forward method runs successfully with valid inputs."""
    num_nodes = 4
    num_edges = 3

    h = torch.rand(num_nodes, 5)  # Node features with 5 dimensions
    x = torch.rand(num_nodes, 3)  # Random 3D coordinates for 4 nodes
    e = torch.randint(0, num_nodes, (2, num_edges))  # Edge connectivity

    new_h, new_x = egcl_instance(h, x, e)

    # Check output dimensions
    assert new_x.shape == x.shape
    assert new_h.shape == h.shape


def test_update_coordinates_only():
    """Test EGCL with only `invariant_update` disabled."""
    egcl = EGCL(dim_h=5, invariant_update=False)

    num_nodes = 3
    h = torch.rand(num_nodes, 5)
    x = torch.rand(num_nodes, 3)
    e = torch.tensor([[0, 1], [1, 2]])

    new_h, new_x = egcl(h, x, e)

    # Features should remain unchanged
    assert torch.allclose(new_h, h)
    # Coordinates should be updated
    assert not torch.allclose(new_x, x)


def test_update_features_only():
    """Test EGCL with only `equivariant_update` disabled."""
    egcl = EGCL(dim_h=5, equivariant_update=False)

    num_nodes = 3
    h = torch.rand(num_nodes, 5)
    x = torch.rand(num_nodes, 3)
    e = torch.tensor([[0, 1], [1, 2]])

    new_h, new_x = egcl(h, x, e)

    # Features should be updated
    assert not torch.allclose(new_h, h)
    # Coordinates should remain unchanged
    assert torch.allclose(new_x, x)


def test_empty_input(egcl_instance):
    """Test EGCL with empty input tensors."""
    h = torch.empty(0, 5)
    x = torch.empty(0, 3)
    e = torch.empty(2, 0, dtype=torch.long)

    new_h, new_x = egcl_instance(h, x, e)

    # Check if outputs are empty as well
    assert new_h.shape == h.shape
    assert new_x.shape == x.shape


def test_invalid_edges(egcl_instance):
    """Test EGCL with invalid edges."""
    h = torch.rand(4, 5)
    x = torch.rand(4, 3)
    e = torch.tensor([[0, 1], [1, 8]])  # Invalid index 8, only 4 nodes

    with pytest.raises(IndexError):
        egcl_instance(h, x, e)


def test_egcl_large_inputs(egcl_instance):
    """Test the numerical stability of EGCL with large input values."""
    num_nodes = 4
    num_edges = 3

    # Input tensors with very large values
    h = torch.full((num_nodes, 5), 1e6, dtype=torch.float32)
    x = torch.full((num_nodes, 3), 1e6, dtype=torch.float32)
    e = torch.randint(0, num_nodes, (2, num_edges))

    new_h, new_x = egcl_instance(h, x, e)

    # Ensure outputs are finite
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_x).all()


def test_egcl_small_inputs(egcl_instance):
    """Test the numerical stability of EGCL with small input values near zero."""
    num_nodes = 4
    num_edges = 3

    # Input tensors with very small values
    h = torch.full((num_nodes, 5), 1e-6, dtype=torch.float32)
    x = torch.full((num_nodes, 3), 1e-6, dtype=torch.float32)
    e = torch.randint(0, num_nodes, (2, num_edges))

    new_h, new_x = egcl_instance(h, x, e)

    # Ensure outputs are finite
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_x).all()


def test_egcl_zero_inputs(egcl_instance):
    """Test the numerical stability of EGCL when all inputs are zero."""
    num_nodes = 4
    num_edges = 3

    # Input tensors with very small values
    h = torch.zeros((num_nodes, 5), dtype=torch.float32)
    x = torch.zeros((num_nodes, 3), dtype=torch.float32)
    e = torch.randint(0, num_nodes, (2, num_edges))

    new_h, new_x = egcl_instance(h, x, e)

    # Ensure outputs are finite
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_x).all()


def test_egcl_zero_distances(egcl_instance):
    """Test the numerical stability of EGCL when edge distances are zero."""
    num_nodes = 4

    # Identical coordinates for all nodes to create zero edge distances
    h = torch.rand((num_nodes, 5), dtype=torch.float32)
    x = torch.zeros((num_nodes, 3), dtype=torch.float32)
    # Self-loops creating edges with zero distances
    e = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])

    new_h, new_x = egcl_instance(h, x, e)

    # Ensure outputs are finite
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_x).all()
