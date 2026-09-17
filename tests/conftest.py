# Standard library
import os
from pathlib import Path

# Third-party
import pytest

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES

# Local
from .dummy_datastore import DummyDatastore

# Disable weights and biases to avoid unnecessary logging
# and to avoid having to deal with authentication
os.environ["WANDB_MODE"] = "disabled"

DATASTORES[DummyDatastore.SHORT_NAME] = DummyDatastore


@pytest.fixture
def datastore():
    """24×24 interior grid with a 12-pixel surface band."""
    n = DummyDatastore.GRID_SIZE
    return DummyDatastore(n_grid_points=n * n, n_timesteps=16)


@pytest.fixture
def datastore_boundary(datastore):
    """28×28 halo grid with 14- and 8-pixel bands for lateral forcing."""
    return DummyDatastore.lateral_boundary(datastore)


def create_test_graph(
    datastore,
    name="1level",
    graph_type="multiscale",
    n_max_levels=1,
    graph_dir_path=None,
    datastore_boundary=None,
):
    """Build a graph under graphs/<name> (or graph_dir_path if given)."""
    if graph_dir_path is None:
        graph_dir_path = datastore.root_path / "graphs" / name
    graph_dir_path = Path(graph_dir_path)
    graph_dir_path.mkdir(parents=True, exist_ok=True)
    create_graph_from_datastore(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=None,
        use_atmosphere_g2m=False,
        graph_dir_path=str(graph_dir_path),
        g2m_radius=0.67,
        g2m_radius_boundary=0.67,
        g2m_radius_atm=0.67,
        mesh_node_distance=25_000.0,
        mesh_refinement_factor=4,
        grid_to_first_mesh_refinement=4,
        n_max_levels=n_max_levels,
        graph_type=graph_type,
        create_plot=False,
        connect_disconnected=True,
        g2m_mean_degree=None,
    )
    return graph_dir_path


class DummyModelArgs:
    """Minimal argparse-like namespace for constructing models in tests."""

    output_std = False
    loss = "mse"
    restore_opt = False
    n_example_pred = 0
    graph = "1level"
    hidden_dim = 4
    hidden_layers = 1
    processor_layers = 1
    mesh_aggr = "sum"
    lr = 1.0e-3
    val_steps_to_log = [1]
    metrics_watch = []
    var_leads_metrics_watch = {}
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    input_steps = 2
    hidden_dim_grid = None
    hidden_dim_edge = None
    hidden_dim_mesh_nodes = None
    grad_checkpointing = False
    use_atmosphere_g2m = False
    num_past_boundary_steps = 1
    num_future_boundary_steps = 0
    current_boundary_step = 1
    num_past_atmosphere_steps = 1
    num_future_atmosphere_steps = 0
    current_atmosphere_step = 1
