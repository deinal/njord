# Third-party
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# First-party
from neural_lam import config as nlconfig
from neural_lam.config import DensityChannel
from neural_lam.models.graphcast import GraphCast
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import DummyModelArgs, create_test_graph
from tests.dummy_datastore import DummyDatastore


def test_dataset_item_shapes(datastore):
    """Validate tensor shapes of a WeatherDataset training sample.

    init_states: (2, N_grid, d_features)
    target_states: (ar_steps, N_grid, d_features)
    forcing: (ar_steps, N_grid, d_windowed_forcing)

    N_grid is the number of surface-sea points, not the full lat/lon grid.
    """
    n_sea = int(
        datastore.get_mask(surface=True, stacked=True, invert=False).sum()
    )

    N_pred_steps = 4
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        split="train",
        ar_steps=N_pred_steps,
        input_steps=2,
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
    )

    item = dataset[0]

    # unpack the item, this is the current return signature for
    # WeatherDataset.__getitem__
    (
        init_states,
        target_states,
        forcing,
        _boundary,
        _atmosphere,
        target_times,
    ) = item

    # initial states
    assert init_states.ndim == 3
    assert init_states.shape[0] == 2  # two time steps go into the input
    assert init_states.shape[1] == n_sea
    assert init_states.shape[2] == datastore.get_num_data_vars("state")

    # output states
    assert target_states.ndim == 3
    assert target_states.shape[0] == N_pred_steps
    assert target_states.shape[1] == n_sea
    assert target_states.shape[2] == datastore.get_num_data_vars("state")

    # forcing
    assert forcing.ndim == 3
    assert forcing.shape[0] == N_pred_steps
    assert forcing.shape[1] == n_sea
    assert forcing.shape[2] == datastore.get_num_data_vars("forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )

    # batch times
    assert target_times.ndim == 1
    assert target_times.shape[0] == N_pred_steps

    # try to get the last item of the dataset to ensure slicing and stacking
    # operations are working as expected and are consistent with the dataset
    # length
    dataset[len(dataset) - 1]


def test_dataset_item_create_dataarray_from_tensor(datastore):
    N_pred_steps = 4
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        split="train",
        ar_steps=N_pred_steps,
        input_steps=2,
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
    )

    idx = 0

    # unpack the item, this is the current return signature for
    # WeatherDataset.__getitem__
    _, target_states, _, _, _, target_times_arr = dataset[idx]
    (
        _,
        da_target_true,
        _,
        _,
        _,
        da_target_times_true,
    ) = dataset._build_item_dataarrays(idx=idx)

    target_times = target_times_arr.numpy().astype("datetime64[ns]")
    np.testing.assert_equal(target_times, da_target_times_true.values)

    da_target = dataset.create_dataarray_from_tensor(
        tensor=target_states, category="state", time=target_times
    )

    land = dataset.land_mask  # True on land
    wet = ~land
    # conversion to torch.float32 may lead to loss of precision
    np.testing.assert_allclose(
        da_target.values[:, wet],
        da_target_true.values[:, wet],
        rtol=1e-6,
    )
    assert np.isnan(da_target.values[:, land]).all()
    assert da_target.dims == da_target_true.dims
    for dim in da_target.dims:
        np.testing.assert_equal(
            da_target[dim].values, da_target_true[dim].values
        )

    da_target_unstacked = datastore.unstack_grid_coords(da_target)
    assert all(
        coord_name in da_target_unstacked.coords for coord_name in ["x", "y"]
    )

    # check construction of a single time
    da_target_single = dataset.create_dataarray_from_tensor(
        tensor=target_states[0], category="state", time=target_times[0]
    )

    # check that the content is the same
    # conversion to torch.float32 may lead to loss of precision
    np.testing.assert_allclose(
        da_target_single.values[wet],
        da_target_true[0].values[wet],
        rtol=1e-6,
    )
    assert np.isnan(da_target_single.values[land]).all()
    assert da_target_single.dims == da_target_true[0].dims
    for dim in da_target_single.dims:
        np.testing.assert_equal(
            da_target_single[dim].values, da_target_true[0][dim].values
        )

    da_target_single_unstacked = datastore.unstack_grid_coords(da_target_single)
    assert all(
        coord_name in da_target_single_unstacked.coords
        for coord_name in ["x", "y"]
    )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_single_batch(datastore, split):
    """Run one GraphCast batch on a WeatherDataset split."""
    device_name = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    graph_name = "1level"

    args = DummyModelArgs()
    args.graph = graph_name

    create_test_graph(datastore, name=graph_name)

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )

    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        split=split,
        ar_steps=2,
        input_steps=2,
    )

    model = GraphCast(
        args=args,
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        config=config,
    )

    model_device = model.to(device_name)
    data_loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(data_loader))
    batch_device = [part.to(device_name) for part in batch]
    model_device.common_step(batch_device)
    model_device.training_step(batch_device)


def test_dataset_item_shapes_with_boundary(datastore, datastore_boundary):
    """Lateral boundary tensor is sea points on the 28×28 halo, windowed."""
    n_sea = int(
        datastore.get_mask(surface=True, stacked=True, invert=False).sum()
    )
    n_sea_boundary = int(
        datastore_boundary.get_mask(
            surface=True, stacked=True, invert=False
        ).sum()
    )
    n_pred_steps = 4
    num_past_boundary_steps = 1
    num_future_boundary_steps = 0
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=None,
        split="train",
        ar_steps=n_pred_steps,
        input_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_boundary_steps=num_past_boundary_steps,
        num_future_boundary_steps=num_future_boundary_steps,
        current_boundary_step=True,
    )
    (
        init_states,
        target_states,
        _forcing,
        boundary,
        _atmosphere,
        _times,
    ) = dataset[0]

    assert init_states.shape[1] == n_sea
    assert target_states.shape[1] == n_sea
    n_bnd_feat = datastore_boundary.get_num_data_vars("forcing")
    n_windows = num_past_boundary_steps + num_future_boundary_steps + 1
    assert boundary.shape == (
        n_pred_steps,
        n_sea_boundary,
        n_bnd_feat * n_windows,
    )
    assert n_sea_boundary == (
        DummyDatastore.BOUNDARY_GRID_SIZE
        * DummyDatastore.BOUNDARY_SURFACE_BAND_WIDTH
    )


def test_single_batch_with_boundary(datastore, datastore_boundary):
    """One GraphCast step with lateral boundary g2m nodes."""
    device_name = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    graph_name = "1level"
    args = DummyModelArgs()
    args.graph = graph_name
    args.num_past_boundary_steps = 1
    args.num_future_boundary_steps = 0
    args.current_boundary_step = 1

    create_test_graph(
        datastore,
        name=graph_name,
        datastore_boundary=datastore_boundary,
    )

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=None,
        split="train",
        ar_steps=2,
        input_steps=2,
        num_past_boundary_steps=1,
        num_future_boundary_steps=0,
        current_boundary_step=True,
    )
    model = GraphCast(
        args=args,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=None,
        config=config,
    )
    model_device = model.to(device_name)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    batch_device = [part.to(device_name) for part in batch]
    model_device.common_step(batch_device)


def test_dataset_item_shapes_with_atmosphere(datastore, datastore_atmosphere):
    """Atmosphere concat uses ocean sea points, not a separate grid."""
    n_sea = int(
        datastore.get_mask(surface=True, stacked=True, invert=False).sum()
    )
    n_pred_steps = 4
    num_past_atmosphere_steps = 1
    num_future_atmosphere_steps = 0
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=datastore_atmosphere,
        split="train",
        ar_steps=n_pred_steps,
        input_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_atmosphere_steps=num_past_atmosphere_steps,
        num_future_atmosphere_steps=num_future_atmosphere_steps,
        current_atmosphere_step=True,
        use_atmosphere_g2m=False,
    )
    (
        init_states,
        _target_states,
        _forcing,
        _boundary,
        atmosphere,
        _times,
    ) = dataset[0]

    n_atm_feat = datastore_atmosphere.get_num_data_vars("forcing")
    n_windows = num_past_atmosphere_steps + num_future_atmosphere_steps + 1
    assert init_states.shape[1] == n_sea
    assert atmosphere.shape == (
        n_pred_steps,
        n_sea,
        n_atm_feat * n_windows,
    )


def test_single_batch_with_atmosphere(datastore, datastore_atmosphere):
    """One GraphCast step with atmosphere concatenated onto interior nodes."""
    device_name = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    graph_name = "1level"
    args = DummyModelArgs()
    args.graph = graph_name
    args.use_atmosphere_g2m = False
    args.num_past_atmosphere_steps = 1
    args.num_future_atmosphere_steps = 0
    args.current_atmosphere_step = 1

    create_test_graph(datastore, name=graph_name)

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=datastore_atmosphere,
        split="train",
        ar_steps=2,
        input_steps=2,
        num_past_atmosphere_steps=1,
        num_future_atmosphere_steps=0,
        current_atmosphere_step=True,
        use_atmosphere_g2m=False,
    )
    model = GraphCast(
        args=args,
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=datastore_atmosphere,
        config=config,
    )
    assert model.concat_atmosphere
    assert model.num_atmosphere_nodes == 0
    model_device = model.to(device_name)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    batch_device = [part.to(device_name) for part in batch]
    model_device.common_step(batch_device)


def test_single_batch_with_boundary_and_atmosphere(
    datastore, datastore_boundary, datastore_atmosphere
):
    """Regional step with lateral g2m boundary plus concatenated atmosphere."""
    device_name = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    graph_name = "1level"
    args = DummyModelArgs()
    args.graph = graph_name
    args.use_atmosphere_g2m = False
    args.num_past_boundary_steps = 1
    args.num_future_boundary_steps = 0
    args.current_boundary_step = 1
    args.num_past_atmosphere_steps = 1
    args.num_future_atmosphere_steps = 0
    args.current_atmosphere_step = 1

    create_test_graph(
        datastore,
        name=graph_name,
        datastore_boundary=datastore_boundary,
    )

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        split="train",
        ar_steps=2,
        input_steps=2,
        num_past_boundary_steps=1,
        num_future_boundary_steps=0,
        current_boundary_step=True,
        num_past_atmosphere_steps=1,
        num_future_atmosphere_steps=0,
        current_atmosphere_step=True,
        use_atmosphere_g2m=False,
    )
    model = GraphCast(
        args=args,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        config=config,
    )
    assert model.boundary_forced
    assert model.concat_atmosphere
    model_device = model.to(device_name)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    batch_device = [part.to(device_name) for part in batch]
    model_device.common_step(batch_device)


def test_dataset_item_shapes_with_atmosphere_g2m(
    datastore, datastore_atmosphere_g2m
):
    """Atmosphere g2m keeps its own 12×12 nodes."""
    n_sea = int(
        datastore.get_mask(surface=True, stacked=True, invert=False).sum()
    )
    n_atm = int(
        datastore_atmosphere_g2m.get_atmosphere_mask(
            stacked=True, invert=False
        ).sum()
    )
    assert n_atm == (
        DummyDatastore.ATMOSPHERE_G2M_GRID_SIZE
        * DummyDatastore.ATMOSPHERE_G2M_GRID_SIZE
    )
    assert n_atm != n_sea
    n_pred_steps = 4
    num_past_atmosphere_steps = 1
    num_future_atmosphere_steps = 0
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=datastore_atmosphere_g2m,
        split="train",
        ar_steps=n_pred_steps,
        input_steps=2,
        num_past_atmosphere_steps=num_past_atmosphere_steps,
        num_future_atmosphere_steps=num_future_atmosphere_steps,
        current_atmosphere_step=True,
        use_atmosphere_g2m=True,
    )
    (
        init_states,
        _target_states,
        _forcing,
        _boundary,
        atmosphere,
        _times,
    ) = dataset[0]
    n_atm_feat = datastore_atmosphere_g2m.get_num_data_vars("forcing")
    n_windows = num_past_atmosphere_steps + num_future_atmosphere_steps + 1
    assert init_states.shape[1] == n_sea
    assert atmosphere.shape == (n_pred_steps, n_atm, n_atm_feat * n_windows)


def test_single_batch_with_atmosphere_g2m(datastore, datastore_atmosphere_g2m):
    """GraphCast step with atmosphere as extra g2m nodes."""
    device_name = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    graph_name = "1level"
    args = DummyModelArgs()
    args.graph = graph_name
    args.use_atmosphere_g2m = True
    args.num_past_atmosphere_steps = 1
    args.num_future_atmosphere_steps = 0
    args.current_atmosphere_step = 1

    create_test_graph(
        datastore,
        name=graph_name,
        datastore_atmosphere=datastore_atmosphere_g2m,
        use_atmosphere_g2m=True,
    )
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=datastore_atmosphere_g2m,
        split="train",
        ar_steps=2,
        input_steps=2,
        num_past_atmosphere_steps=1,
        num_future_atmosphere_steps=0,
        current_atmosphere_step=True,
        use_atmosphere_g2m=True,
    )
    model = GraphCast(
        args=args,
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=datastore_atmosphere_g2m,
        config=config,
    )
    n_sea = int(
        datastore.get_mask(surface=True, stacked=True, invert=False).sum()
    )
    n_atm = int(
        datastore_atmosphere_g2m.get_atmosphere_mask(
            stacked=True, invert=False
        ).sum()
    )
    assert model.use_atmosphere_g2m
    assert not model.concat_atmosphere
    assert model.num_atmosphere_nodes == n_atm
    assert model.num_total_grid_nodes == n_sea + n_atm
    assert int(model.g2m_edge_index[0].max()) >= n_sea
    model_device = model.to(device_name)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    batch_device = [part.to(device_name) for part in batch]
    model_device.common_step(batch_device)


def test_lat_weighted_loss(datastore):
    """Loss weights follow cos(get_xy lat column); dry 50 m cells stay 0."""
    graph_name = "1level"
    create_test_graph(datastore, name=graph_name)
    args = DummyModelArgs()
    args.graph = graph_name
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        ),
        training=nlconfig.TrainingConfig(lat_weighted_loss=True),
    )
    model = GraphCast(
        args=args,
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        config=config,
    )
    names = datastore.get_vars_names("state")
    deep_idx = names.index("thetao_50m")
    surface = datastore.get_mask(surface=True, stacked=True, invert=False)
    xy_surface = datastore.get_xy("state", stacked=True)[surface]
    cos_lat = np.cos(np.deg2rad(xy_surface[:, 1].astype(np.float64))).astype(
        np.float32
    )
    interior = model.interior_mask.astype(np.float32)
    expected = interior * cos_lat[:, np.newaxis]
    expected *= interior.sum() / expected.sum()
    np.testing.assert_allclose(model.loss_mask.numpy(), expected, rtol=1e-5)
    active = model.interior_mask_bool
    weights = model.loss_mask
    assert torch.all(weights[~active] == 0)
    assert torch.isclose(weights[active].mean(), torch.tensor(1.0), atol=1e-5)
    dry_deep = ~active[:, deep_idx]
    assert dry_deep.any()
    assert torch.all(weights[dry_deep, deep_idx] == 0)


@pytest.mark.parametrize(
    "dataset_config",
    [
        {"past": 0, "future": 0, "ar_steps": 1, "exp_len_reduction": 3},
        {"past": 2, "future": 0, "ar_steps": 1, "exp_len_reduction": 3},
        {"past": 0, "future": 2, "ar_steps": 1, "exp_len_reduction": 5},
        {"past": 4, "future": 0, "ar_steps": 1, "exp_len_reduction": 5},
        {"past": 0, "future": 0, "ar_steps": 5, "exp_len_reduction": 7},
        {"past": 3, "future": 3, "ar_steps": 2, "exp_len_reduction": 8},
    ],
)
def test_dataset_length(dataset_config):
    """Check that correct number of samples can be extracted from the dataset,
    given a specific configuration of forcing windowing and ar_steps.
    """
    # Use dummy datastore of length 10 here, only want to test slicing
    # in dataset class
    ds_len = 10
    datastore = DummyDatastore(n_grid_points=64, n_timesteps=ds_len)

    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        split="train",
        ar_steps=dataset_config["ar_steps"],
        input_steps=2,
        num_past_forcing_steps=dataset_config["past"],
        num_future_forcing_steps=dataset_config["future"],
    )

    # We expect dataset to contain this many samples
    expected_len = ds_len - dataset_config["exp_len_reduction"]

    # Check that datast has correct length
    assert len(dataset) == expected_len

    # Check that we can actually get last and first sample
    dataset[0]
    dataset[expected_len - 1]


def test_dataset_zeros_below_bathymetry(datastore):
    """Land and below-bottom cells are zeroed on the sea-point tensor."""
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        split="train",
        ar_steps=2,
        input_steps=2,
        standardize=False,
    )
    init_states, target_states, *_ = dataset[0]
    dry = torch.as_tensor(dataset.land_mask_bool, dtype=torch.bool)
    assert dry.any(), "expected some dry cells at depth"
    assert torch.all(init_states[:, dry[0]] == 0)
    assert torch.all(target_states[:, dry[0]] == 0)
    # surface features stay wet on every sea point
    names = datastore.get_vars_names("state")
    zos_idx = names.index("zos")
    assert not dry[0, :, zos_idx].any()


def test_boundary_zeros_dry_windowed_features(datastore, datastore_boundary):
    """Halo forcing stays 0 at dry 50m cells after the time window is stacked"""
    datastore_boundary.fill_constant("forcing", 7.0)
    names = datastore_boundary.get_vars_names("forcing")
    n_feat = len(names)
    n_windows = 2
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=None,
        split="train",
        ar_steps=2,
        input_steps=2,
        standardize=False,
        num_past_boundary_steps=1,
        num_future_boundary_steps=0,
        current_boundary_step=True,
    )
    _init, _target, _forcing, boundary, *_ = dataset[0]
    dry = torch.as_tensor(dataset.land_mask_bool_boundary[0], dtype=torch.bool)
    deep_idx = names.index("thetao_50m")
    surf_idx = names.index("zos")
    assert dry[:, deep_idx].any(), "expected dry 50 m cells on the halo"
    assert not dry[:, surf_idx].any()

    deep_cols = slice(deep_idx * n_windows, (deep_idx + 1) * n_windows)
    surf_cols = slice(surf_idx * n_windows, (surf_idx + 1) * n_windows)
    assert torch.all(boundary[:, dry[:, deep_idx], deep_cols] == 0)
    assert torch.all(boundary[:, ~dry[:, surf_idx], surf_cols] == 7)
    assert boundary.shape[-1] == n_feat * n_windows


def test_density_channel(datastore):
    """Density follows siconc > 0; sithick does not create ice on its own."""
    names = datastore.get_vars_names("state")
    siconc_idx = names.index("siconc")
    sithick_idx = names.index("sithick")
    da = datastore.ds["state"]
    feat_axis = da.get_axis_num("state_feature")
    grid_axis = da.get_axis_num("grid_index")
    surface = datastore.get_mask(surface=True, stacked=True, invert=False)
    sea = np.where(surface)[0]
    assert sea.size >= 2

    slc = [slice(None)] * da.ndim
    slc[feat_axis] = siconc_idx
    da.values[tuple(slc)] = 0.0
    slc[grid_axis] = sea[0]
    da.values[tuple(slc)] = 0.8

    slc = [slice(None)] * da.ndim
    slc[feat_axis] = sithick_idx
    da.values[tuple(slc)] = 0.0
    slc[grid_axis] = sea[:2]
    da.values[tuple(slc)] = 1.5

    density_cfg = DensityChannel(
        reference_var="siconc", associated_vars=["siconc", "sithick"]
    )
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        split="train",
        ar_steps=2,
        input_steps=2,
        standardize=False,
        density_channel=density_cfg,
    )
    init_states, target_states, *_ = dataset[0]
    n_state = datastore.get_num_data_vars("state")
    assert init_states.shape[-1] == n_state + 1
    density = init_states[:, :, -1]
    siconc = init_states[:, :, siconc_idx]
    sithick = init_states[:, :, sithick_idx]
    assert torch.all(density[siconc > 0] == 1)
    assert torch.all(density[siconc <= 0] == 0)
    assert (sithick > 0).any()
    assert torch.all(density[sithick > 0] == (siconc[sithick > 0] > 0).float())
    assert torch.all(
        target_states[:, :, -1] == (target_states[:, :, siconc_idx] > 0).float()
    )

    graph_name = "1level"
    create_test_graph(datastore, name=graph_name)
    args = DummyModelArgs()
    args.graph = graph_name
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        ),
        training=nlconfig.TrainingConfig(density_channel=density_cfg),
    )
    model = GraphCast(
        args=args,
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        config=config,
    )
    n_sea = init_states.shape[1]
    state = torch.full((1, n_sea, n_state + 1), 0.5)
    ice = torch.zeros(n_sea, dtype=torch.bool)
    ice[0] = True
    state[:, :, -1] = -4.0
    state[:, ice, -1] = 4.0
    out = model.apply_density_threshold(state)
    ice_val = (1.0 - model.state_mean[-1]) / model.state_std[-1]
    no_ice = -model.state_mean / model.state_std
    assert torch.allclose(out[:, ice, -1], ice_val)
    assert torch.allclose(out[:, ~ice, -1], no_ice[-1])
    assert torch.allclose(out[:, ice, siconc_idx], state[:, ice, siconc_idx])
    assert torch.allclose(out[:, ice, sithick_idx], state[:, ice, sithick_idx])
    assert torch.allclose(out[:, ~ice, siconc_idx], no_ice[siconc_idx])
    assert torch.allclose(out[:, ~ice, sithick_idx], no_ice[sithick_idx])


def test_interior_loss_mask_at_depth(datastore):
    """Loss on the 12-pixel surface band, at 50 m only the 6-pixel subset."""
    graph_name = "1level"
    create_test_graph(datastore, name=graph_name)
    args = DummyModelArgs()
    args.graph = graph_name
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )
    model = GraphCast(
        args=args,
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        config=config,
    )
    names = datastore.get_vars_names("state")
    surf_idx = names.index("zos")
    deep_idx = names.index("thetao_50m")
    n_sea = int(
        datastore.get_mask(surface=True, stacked=True, invert=False).sum()
    )
    n_deep = DummyDatastore.GRID_SIZE * DummyDatastore.DEPTH_BAND_WIDTH
    assert model.interior_mask.shape[0] == n_sea
    assert model.interior_mask[:, surf_idx].all()
    assert int(model.interior_mask[:, deep_idx].sum()) == n_deep
    assert not model.interior_mask[:, deep_idx].all()
    dry_deep = ~model.interior_mask_bool[:, deep_idx]
    assert torch.all(model.loss_mask[dry_deep, deep_idx] == 0)
    assert torch.all(model.loss_mask[:, surf_idx] == 1)
