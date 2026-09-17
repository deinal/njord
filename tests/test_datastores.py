"""Checks that DummyDatastore implements the BaseRegularGridDatastore API."""

# Standard library
import collections
import dataclasses
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import numpy as np
import pytest
import torch
import xarray as xr

# First-party
from neural_lam.datastore.plot_example import plot_example_from_datastore
from tests.dummy_datastore import DummyDatastore


def test_root_path(datastore):
    """Check that the `datastore.root_path` property is implemented."""
    assert isinstance(datastore.root_path, Path)


def test_config(datastore):
    """Check that the `datastore.config` property is implemented."""
    config = datastore.config
    assert isinstance(
        config, collections.abc.Mapping
    ) or dataclasses.is_dataclass(config)


def test_step_length(datastore):
    """Check that the `datastore.step_length` property is implemented."""
    step_length = datastore.step_length
    assert isinstance(step_length, int)
    assert step_length > 0


def test_datastore_grid_xy(datastore):
    """Check `get_xy` shape against `grid_shape_state`."""
    grid_shape = datastore.grid_shape_state
    nx, ny = grid_shape.x, grid_shape.y
    for stacked in [True, False]:
        xy = datastore.get_xy("static", stacked=stacked)
        if stacked:
            assert xy.shape == (nx * ny, 2)
        else:
            assert xy.shape == (nx, ny, 2)


def test_get_vars(datastore):
    """Check that results of.

    - `datastore.get_vars_units`
    - `datastore.get_vars_names`
    - `datastore.get_vars_long_names`
    - `datastore.get_num_data_vars`

    are consistent (as in the number of variables are the same) and that the
    return types of each are correct.

    """
    for category in ["state", "forcing", "static"]:
        units = datastore.get_vars_units(category)
        names = datastore.get_vars_names(category)
        long_names = datastore.get_vars_long_names(category)
        num_vars = datastore.get_num_data_vars(category)

        assert len(units) == len(names) == num_vars
        assert isinstance(units, list)
        assert isinstance(names, list)
        assert isinstance(long_names, list)
        assert isinstance(num_vars, int)


def test_get_normalization_dataarray(datastore):
    """Check that the `datastore.get_normalization_dataarray` method is
    implemented."""
    for category in ["state", "forcing", "static"]:
        ds_stats = datastore.get_standardization_dataarray(category=category)

        # check that the returned object is an xarray DataArray
        # and that it has the correct variables
        assert isinstance(ds_stats, xr.Dataset)

        if category == "state":
            ops = [
                "mean",
                "std",
                "diff_mean_standardized",
                "diff_std_standardized",
            ]
        elif category == "forcing":
            ops = ["mean", "std"]
        elif category == "static":
            ops = []
        else:
            raise NotImplementedError(category)

        for op in ops:
            var_name = f"{category}_{op}"
            assert var_name in ds_stats.data_vars
            da_val = ds_stats[var_name]
            assert set(da_val.dims) == {f"{category}_feature"}


def test_get_dataarray(datastore):
    """Check that the `datastore.get_dataarray` method is implemented.

    And that it returns an xarray DataArray with the correct dimensions.

    """
    for category in ["state", "forcing", "static"]:
        n_features = {}
        if category in ["state", "forcing"]:
            splits = ["train", "val", "test"]
        elif category == "static":
            # static data should be the same for all splits, so split
            # should be allowed to be None
            splits = ["train", "val", "test", None]
        else:
            raise NotImplementedError(category)

        for split in splits:
            expected_dims = ["grid_index", f"{category}_feature"]
            if category != "static":
                if not datastore.is_forecast:
                    expected_dims.append("time")
                else:
                    expected_dims += [
                        "init_time",
                        "lead_time",
                    ]

            if datastore.is_ensemble and category == "state":
                # assume that only state variables change with ensemble members
                expected_dims.append("ensemble_member")

            da = datastore.get_dataarray(category=category, split=split)

            assert isinstance(da, xr.DataArray)
            assert set(da.dims) == set(expected_dims)
            grid_shape = datastore.grid_shape_state
            assert da.grid_index.size == grid_shape.x * grid_shape.y

            n_features[split] = da[category + "_feature"].size

        # check that the number of features is the same for all splits
        assert n_features["train"] == n_features["val"] == n_features["test"]


def test_boundary_mask(datastore):
    """Check that the `datastore.boundary_mask` property is implemented and
    that the returned object is an xarray DataArray with the correct shape."""
    da_mask = datastore.boundary_mask

    assert isinstance(da_mask, xr.DataArray)
    assert set(da_mask.dims) == {"grid_index"}
    assert da_mask.dtype == "int"
    assert set(da_mask.values) == {0, 1}
    assert da_mask.sum() > 0
    assert da_mask.sum() < da_mask.size

    grid_shape = datastore.grid_shape_state
    assert datastore.boundary_mask.size == grid_shape.x * grid_shape.y


def test_ocean_mask_nested_depths(datastore):
    """Surface band is 12 pixels; 50 m band is 6 pixels nested inside it."""
    n_x, n_y = datastore.grid_shape_state.x, datastore.grid_shape_state.y
    n_grid = n_x * n_y
    n_feat = datastore.get_num_data_vars("state")
    names = datastore.get_vars_names("state")

    assert (n_x, n_y) == (DummyDatastore.GRID_SIZE, DummyDatastore.GRID_SIZE)
    assert list(datastore.ds.depth.values) == list(DummyDatastore.DEPTHS_M)
    assert "thetao_0m" in names
    assert "thetao_50m" in names

    surface = datastore.get_mask(surface=True, stacked=True, invert=False)
    full = datastore.get_mask(surface=False, stacked=True, invert=False)
    surface_xy = datastore.get_mask(surface=True, stacked=False, invert=False)
    full_xy = datastore.get_mask(surface=False, stacked=False, invert=False)

    assert surface.dtype == bool
    assert surface.shape == (n_grid,)
    assert not surface.all()
    assert surface.any()
    assert surface_xy.shape == (n_x, n_y)
    np.testing.assert_array_equal(surface, surface_xy.reshape(-1))
    assert surface_xy.sum() == n_x * DummyDatastore.SURFACE_BAND_WIDTH
    assert int(surface_xy.any(axis=1).sum()) == n_x
    assert (
        int(surface_xy.all(axis=0).sum()) == DummyDatastore.SURFACE_BAND_WIDTH
    )

    assert full.shape == (n_grid, n_feat)
    assert full_xy.shape == (n_x, n_y, n_feat)
    np.testing.assert_array_equal(full[:, 0], surface)

    n_wet = full.sum(axis=0)
    deep_idx = names.index("thetao_50m")
    surface_idx = names.index("thetao_0m")
    assert n_wet[deep_idx] == n_x * DummyDatastore.DEPTH_BAND_WIDTH
    assert n_wet[deep_idx] < n_wet[surface_idx]
    np.testing.assert_array_equal(
        full[:, deep_idx], full[:, deep_idx] & full[:, surface_idx]
    )
    assert np.all(surface[full[:, deep_idx]])

    inverted = datastore.get_mask(surface=True, stacked=True, invert=True)
    np.testing.assert_array_equal(inverted, ~surface)


def test_lateral_boundary_halo_bands(datastore, datastore_boundary):
    """28×28 halo: 14-pixel surface band, 8-pixel 50 m band, nested."""
    n_x, n_y = (
        datastore_boundary.grid_shape_state.x,
        datastore_boundary.grid_shape_state.y,
    )
    assert (n_x, n_y) == (
        DummyDatastore.BOUNDARY_GRID_SIZE,
        DummyDatastore.BOUNDARY_GRID_SIZE,
    )
    assert (
        datastore_boundary.surface_band_width
        == DummyDatastore.BOUNDARY_SURFACE_BAND_WIDTH
    )
    assert (
        datastore_boundary.depth_band_width
        == DummyDatastore.BOUNDARY_DEPTH_BAND_WIDTH
    )

    surface_xy = datastore_boundary.get_mask(
        surface=True, stacked=False, invert=False
    )
    full_xy = datastore_boundary.get_mask(
        surface=False, stacked=False, invert=False
    )
    names = datastore_boundary.get_vars_names("state")
    deep_idx = names.index("thetao_50m")

    assert surface_xy.sum() == n_x * DummyDatastore.BOUNDARY_SURFACE_BAND_WIDTH
    assert full_xy[:, :, deep_idx].sum() == (
        n_x * DummyDatastore.BOUNDARY_DEPTH_BAND_WIDTH
    )
    assert np.all(full_xy[:, :, deep_idx] <= surface_xy)
    assert datastore_boundary.get_vars_names("forcing") == names

    halo = DummyDatastore.HALO_CELLS
    interior_xy = datastore.get_mask(surface=True, stacked=False, invert=False)
    interior_on_halo = surface_xy[
        halo : halo + interior_xy.shape[0],
        halo : halo + interior_xy.shape[1],
    ]
    assert np.all(interior_on_halo[interior_xy])


def test_atmosphere_shares_interior_grid(datastore, datastore_atmosphere):
    """Atmosphere concat path requires the same 24×24 grid as the ocean."""
    assert (
        datastore_atmosphere.grid_shape_state.x,
        datastore_atmosphere.grid_shape_state.y,
    ) == (datastore.grid_shape_state.x, datastore.grid_shape_state.y)
    np.testing.assert_allclose(datastore_atmosphere._x, datastore._x)
    np.testing.assert_allclose(datastore_atmosphere._y, datastore._y)
    np.testing.assert_array_equal(
        datastore_atmosphere.get_mask(surface=True, stacked=True, invert=False),
        datastore.get_mask(surface=True, stacked=True, invert=False),
    )
    assert datastore_atmosphere.get_vars_names("forcing") == list(
        DummyDatastore.ATMOSPHERE_FORCING_NAMES
    )


def test_get_xy_extent(datastore):
    """Check that the `datastore.get_xy_extent` method is implemented and that
    the returned object is a tuple of the correct length."""
    extents = {}
    # get the extents for each category, and finally check they are all the same
    for category in ["state", "forcing", "static"]:
        extent = datastore.get_xy_extent(category)
        assert isinstance(extent, list)
        assert len(extent) == 4
        assert all(isinstance(e, (int, float)) for e in extent)
        extents[category] = extent

    # check that the extents are the same for all categories
    for category in ["forcing", "static"]:
        assert extents["state"] == extents[category]


def test_get_xy(datastore):
    """Check that the `datastore.get_xy` method is implemented."""
    for category in ["state", "forcing", "static"]:
        xy_stacked = datastore.get_xy(category=category, stacked=True)
        xy_unstacked = datastore.get_xy(category=category, stacked=False)

        assert isinstance(xy_stacked, np.ndarray)
        assert isinstance(xy_unstacked, np.ndarray)

        nx, ny = datastore.grid_shape_state.x, datastore.grid_shape_state.y

        # for stacked=True, the shape should be (n_grid_points, 2)
        assert xy_stacked.ndim == 2
        assert xy_stacked.shape[0] == nx * ny
        assert xy_stacked.shape[1] == 2

        # for stacked=False, the shape should be (nx, ny, 2)
        assert xy_unstacked.ndim == 3
        assert xy_unstacked.shape[0] == nx
        assert xy_unstacked.shape[1] == ny
        assert xy_unstacked.shape[2] == 2


def test_get_projection(datastore):
    """Check that the `datastore.coords_projection` property is implemented."""
    assert isinstance(datastore.coords_projection, ccrs.Projection)


def test_grid_shape_state(datastore):
    """Check that the `datastore.grid_shape_state` property is implemented."""
    grid_shape = datastore.grid_shape_state
    assert grid_shape.x > 0
    assert grid_shape.y > 0


@pytest.mark.parametrize("category", ["state", "forcing", "static"])
def test_stacking_grid_coords(datastore, category):
    """Check that the `datastore.stack_grid_coords` method is implemented."""
    da_static = datastore.get_dataarray(category=category, split="train")

    da_static_unstacked = datastore.unstack_grid_coords(da_static).load()
    da_static_test = datastore.stack_grid_coords(da_static_unstacked)

    assert da_static.dims == da_static_test.dims
    xr.testing.assert_equal(da_static, da_static_test)


def test_dataarray_shapes(datastore):
    static_da = datastore.get_dataarray("static", split=None)
    static_da = datastore.stack_grid_coords(static_da)
    static_da = static_da.isel(static_feature=0)

    # Convert the unstacked grid coordinates and static data array to tensors
    unstacked_tensor = torch.tensor(
        datastore.unstack_grid_coords(static_da).to_numpy(), dtype=torch.float32
    ).squeeze()

    reshaped_tensor = (
        torch.tensor(static_da.to_numpy(), dtype=torch.float32)
        .reshape(datastore.grid_shape_state.x, datastore.grid_shape_state.y)
        .squeeze()
    )

    # Compute the difference
    diff = unstacked_tensor - reshaped_tensor

    # Check the shapes
    assert unstacked_tensor.shape == (
        datastore.grid_shape_state.x,
        datastore.grid_shape_state.y,
    )
    assert reshaped_tensor.shape == (
        datastore.grid_shape_state.x,
        datastore.grid_shape_state.y,
    )
    assert diff.shape == (
        datastore.grid_shape_state.x,
        datastore.grid_shape_state.y,
    )
    # assert diff == 0 with tolerance 1e-6
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-6)


def test_plot_example_from_datastore(datastore):
    """Check that the `plot_example_from_datastore` function is implemented."""
    fig = plot_example_from_datastore(
        category="static",
        datastore=datastore,
        col_dim="{category}_feature",
        split="train",
        standardize=True,
        selection={},
        index_selection={},
    )

    assert fig is not None
    assert fig.get_axes()


@pytest.mark.parametrize("category", ("state", "static"))
def test_get_standardized_da(datastore, category):
    """Check that dataarray is actually standardized when calling
    get_dataarray with standardize=True"""
    ds_stats = datastore.get_standardization_dataarray(category=category)

    mean = ds_stats[f"{category}_mean"]
    std = ds_stats[f"{category}_std"]

    non_standard_da = datastore.get_dataarray(
        category=category, split="train", standardize=False
    )
    standard_da = datastore.get_dataarray(
        category=category, split="train", standardize=True
    )

    assert np.allclose(standard_da, (non_standard_da - mean) / std, atol=1e-6)
