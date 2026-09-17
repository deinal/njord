# Standard library
import tempfile
from pathlib import Path

# Third-party
import pytest
import torch
from torch.utils.data import DataLoader

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.graph_fm import GraphFM
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import DummyModelArgs, create_test_graph


@pytest.mark.parametrize(
    "graph_name", ["1level", "multiscale", "hierarchical", "cluster"]
)
def test_graph_creation(datastore, graph_name):
    """Check that create_graph_from_datastore writes the expected files."""
    extra = {}
    if graph_name == "hierarchical":
        hierarchical = True
        graph_type = "hierarchical"
        n_max_levels = 2
    elif graph_name == "multiscale":
        hierarchical = False
        graph_type = "multiscale"
        n_max_levels = 2
    elif graph_name == "1level":
        hierarchical = False
        graph_type = "multiscale"
        n_max_levels = 1
    elif graph_name == "cluster":
        hierarchical = True
        graph_type = "cluster"
        n_max_levels = 2
        extra["max_edge_len"] = 250_000.0
    else:
        raise ValueError(f"Unknown graph_name: {graph_name}")

    required_graph_files = [
        "m2m_edge_index.pt",
        "g2m_edge_index.pt",
        "m2g_edge_index.pt",
        "m2m_features.pt",
        "g2m_features.pt",
        "m2g_features.pt",
        "mesh_features.pt",
    ]
    if hierarchical:
        required_graph_files.extend(
            [
                "mesh_up_edge_index.pt",
                "mesh_down_edge_index.pt",
                "mesh_up_features.pt",
                "mesh_down_features.pt",
            ]
        )

    # TODO: check that the number of edges is consistent over the files, for
    # now we just check the number of features
    d_features = 3
    d_mesh_static = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graphs" / graph_name

        create_test_graph(
            datastore=datastore,
            name=graph_name,
            graph_type=graph_type,
            n_max_levels=n_max_levels,
            graph_dir_path=graph_dir_path,
            **extra,
        )

        assert graph_dir_path.exists()

        # check that all the required files are present
        for file_name in required_graph_files:
            assert (graph_dir_path / file_name).exists()

        # try to load each and ensure they have the right shape
        for file_name in required_graph_files:
            file_id = Path(file_name).stem  # remove the extension
            result = torch.load(graph_dir_path / file_name, weights_only=True)

            if file_id.startswith("g2m") or file_id.startswith("m2g"):
                assert isinstance(result, torch.Tensor)

                if file_id.endswith("_index"):
                    assert (
                        result.shape[0] == 2
                    )  # adjacency matrix uses two rows
                elif file_id.endswith("_features"):
                    assert result.shape[1] == d_features

            elif file_id.startswith("m2m") or file_id.startswith("mesh"):
                assert isinstance(result, list)
                if not hierarchical:
                    assert len(result) == 1
                else:
                    if file_id.startswith("mesh_up") or file_id.startswith(
                        "mesh_down"
                    ):
                        assert len(result) == n_max_levels - 1
                    else:
                        assert len(result) == n_max_levels

                for r in result:
                    assert isinstance(r, torch.Tensor)

                    if file_id == "mesh_features":
                        assert r.shape[1] == d_mesh_static
                        assert r.shape[0] > 0
                    elif file_id.endswith("_index"):
                        assert r.shape[0] == 2  # adjacency matrix uses two rows
                        assert r.shape[1] > 0
                    elif file_id.endswith("_features"):
                        assert r.shape[1] == d_features


def test_graph_fm_two_level_hierarchy(datastore):
    """Test the hierarchical Graph-FM model."""
    graph_name = "hierarchical"
    create_test_graph(
        datastore,
        name=graph_name,
        graph_type="hierarchical",
        n_max_levels=2,
    )
    args = DummyModelArgs()
    args.graph = graph_name
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )
    model = GraphFM(
        args=args,
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        config=config,
    )
    assert model.hierarchical
    assert model.num_levels == 2
    assert all(size > 0 for size in model.level_mesh_sizes)

    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        split="train",
        ar_steps=2,
        input_steps=2,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model.common_step(batch)


def test_graph_fm_cluster(datastore):
    """Two-level cluster mesh loads in Graph-FM."""
    graph_name = "cluster"
    create_test_graph(
        datastore,
        name=graph_name,
        graph_type="cluster",
        n_max_levels=2,
        max_edge_len=250_000.0,
    )
    args = DummyModelArgs()
    args.graph = graph_name
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )
    model = GraphFM(
        args=args,
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        config=config,
    )
    assert model.hierarchical
    assert model.num_levels == 2
    assert all(size > 0 for size in model.level_mesh_sizes)
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        split="train",
        ar_steps=2,
        input_steps=2,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model.common_step(batch)
