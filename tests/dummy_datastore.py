# Standard library
import datetime
import tempfile
from functools import cached_property
from pathlib import Path
from typing import List, Union

# Third-party
import isodate
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from numpy import ndarray

# First-party
from neural_lam.datastore.base import (
    BaseRegularGridDatastore,
    CartesianGridShape,
)


class DummyDatastore(BaseRegularGridDatastore):
    """
    Datastore that creates some dummy data for testing purposes. The data
    consists of state, forcing, and static variables, and is stored in a
    regular Lambert Conformal grid over the Baltic Sea (500x500 km, centered
    at 20°E, 60°N).

    Sea/land masks are centered horizontal bands (full ``x``, limited ``y``):
    12 pixels at the surface and 6 pixels at 50 m on the default 24×24
    interior grid. A lateral-boundary datastore is a 28×28 halo with 14- and
    8-pixel bands. State features follow MDP-style stacking, with depth in
    the name (``thetao_0m``, ``thetao_50m``, ...).
    """

    SHORT_NAME = "dummydata"
    T0 = isodate.parse_datetime("2021-01-01T00:00:00")
    GRID_SIZE = 24
    DEPTHS_M = (0, 50)
    SURFACE_BAND_WIDTH = 12
    DEPTH_BAND_WIDTH = 6
    HALO_CELLS = 2
    BOUNDARY_GRID_SIZE = GRID_SIZE + 2 * HALO_CELLS
    BOUNDARY_SURFACE_BAND_WIDTH = 14
    BOUNDARY_DEPTH_BAND_WIDTH = 8
    FEATURE_NAMES = {
        "state": ["zos", "thetao_0m", "so_0m", "thetao_50m", "so_50m"],
        "forcing": ["forcing_feat_0", "forcing_feat_1"],
        "static": ["deptho"],
    }
    N_FEATURES = {k: len(v) for k, v in FEATURE_NAMES.items()}
    DEEP_STATE_FEATURES = frozenset(["thetao_50m", "so_50m"])
    CARTESIAN_COORDS = ["x", "y"]

    bbox_size_km = [500, 500]  # km

    def __init__(
        self,
        config_path=None,
        n_grid_points=None,
        n_timesteps=10,
        surface_band_width=None,
        depth_band_width=None,
        x_coords=None,
        y_coords=None,
        feature_names=None,
    ) -> None:
        """
        Create a dummy datastore with random data.

        Parameters
        ----------
        config_path : None
            No config file is needed for the dummy datastore. This argument is
            only present to match the signature of the other datastores.
        n_grid_points : int
            The number of grid points in the dataset. Must be a perfect square.
            Ignored when ``x_coords`` / ``y_coords`` are given.
        n_timesteps : int
            The number of timesteps in the dataset.
        surface_band_width : int
            Height in grid cells of the surface sea band. Defaults to 12.
        depth_band_width : int
            Height in grid cells of the 50 m sea band. Defaults to 6.
        x_coords, y_coords : array-like
            Optional projected coordinates (m). Used for the 28×28 halo grid.
        feature_names : dict
            Optional override of ``FEATURE_NAMES`` (boundary uses ocean vars
            as forcing).
        """
        assert (
            config_path is None
        ), "No config file is needed for the dummy datastore"

        self._feature_names = feature_names or self.FEATURE_NAMES
        self._n_timesteps = n_timesteps

        if x_coords is not None or y_coords is not None:
            assert x_coords is not None and y_coords is not None
            x = np.asarray(x_coords, dtype=float)
            y = np.asarray(y_coords, dtype=float)
            assert x.ndim == y.ndim == 1
            assert x.size == y.size, "dummy grid must be square"
            n_points_1d = int(x.size)
            n_grid_points = n_points_1d * n_points_1d
        else:
            if n_grid_points is None:
                n_grid_points = self.GRID_SIZE * self.GRID_SIZE
            n_points_1d = int(np.sqrt(n_grid_points))
            assert (
                n_points_1d * n_points_1d == n_grid_points
            ), "n_grid_points must be a perfect square"
            lx, ly = self.bbox_size_km
            x = np.linspace(-lx / 2.0 * 1.0e3, lx / 2.0 * 1.0e3, n_points_1d)
            y = np.linspace(-ly / 2.0 * 1.0e3, ly / 2.0 * 1.0e3, n_points_1d)

        if surface_band_width is None:
            surface_band_width = self.SURFACE_BAND_WIDTH
        if depth_band_width is None:
            depth_band_width = self.DEPTH_BAND_WIDTH
        surface_band_width = min(int(surface_band_width), n_points_1d)
        depth_band_width = min(int(depth_band_width), surface_band_width)
        self.surface_band_width = surface_band_width
        self.depth_band_width = depth_band_width
        self._x = x
        self._y = y

        xs, ys = np.meshgrid(x, y)

        # Create lat/lon coordinates from the Lambert grid
        lon_mesh, lat_mesh = (
            ccrs.PlateCarree()
            .transform_points(
                src_crs=self.coords_projection,
                x=xs.flatten(),
                y=ys.flatten(),
            )[:, :2]
            .T
        )

        # Create base dataset with proper coordinates
        self.ds = xr.Dataset(
            coords={
                "x": (
                    "x",
                    x,
                    {"units": "m"},
                ),  # Use first column for x coordinates
                "y": (
                    "y",
                    y,
                    {"units": "m"},
                ),  # Use first row for y coordinates
                "longitude": (
                    "grid_index",
                    lon_mesh.flatten(),
                    {"units": "degrees_east"},
                ),
                "latitude": (
                    "grid_index",
                    lat_mesh.flatten(),
                    {"units": "degrees_north"},
                ),
                "depth": (
                    "depth",
                    np.array(self.DEPTHS_M, dtype=float),
                    {"units": "m", "positive": "down"},
                ),
            }
        )
        self._surface_band = self._horizontal_band(
            n_points_1d, self.surface_band_width
        )
        self._depth_band = self._horizontal_band(
            n_points_1d, self.depth_band_width
        )

        # Create data variables with proper dimensions
        for category, feature_names in self._feature_names.items():
            n = len(feature_names)
            feature_units = ["-" for _ in range(n)]  # Placeholder units
            feature_long_names = [
                f"Long name for {name}" for name in feature_names
            ]

            self.ds[f"{category}_feature"] = feature_names
            self.ds[f"{category}_feature_units"] = (
                f"{category}_feature",
                feature_units,
            )
            self.ds[f"{category}_feature_long_name"] = (
                f"{category}_feature",
                feature_long_names,
            )

            # Define dimensions and create random data
            dims = ["grid_index", f"{category}_feature"]
            if category != "static":
                dims.append("time")
                shape = (n_grid_points, n, n_timesteps)
            else:
                shape = (n_grid_points, n)

            # Create random data
            data = np.random.randn(*shape)

            # Create DataArray with proper dimensions
            self.ds[category] = xr.DataArray(
                data,
                dims=dims,
                coords={
                    f"{category}_feature": feature_names,
                },
            )

            if category != "static":
                dt = datetime.timedelta(hours=self.step_length)
                times = [self.T0 + dt * i for i in range(n_timesteps)]
                self.ds.coords["time"] = times

        # Add boundary mask
        self.ds["boundary_mask"] = xr.DataArray(
            np.random.choice([0, 1], size=(n_points_1d, n_points_1d)),
            dims=["x", "y"],
        )

        # Stack the spatial dimensions into grid_index
        self.ds = self.ds.stack(grid_index=self.CARTESIAN_COORDS)

        # Create temporary directory for storing derived files
        self._tempdir = tempfile.TemporaryDirectory()
        self._root_path = Path(self._tempdir.name)
        self._num_grid_points = n_grid_points

    @classmethod
    def lateral_boundary(cls, interior: "DummyDatastore", halo_cells=None):
        """28×28 halo grid with a 14-pixel surface band and 8-pixel 50 m band.

        Coordinates keep the interior spacing and extend ``halo_cells`` (2)
        points on each side so the extra wet rows/columns are a lateral halo.
        Forcing features are the same ocean state variables (parent-model
        analogue).
        """
        if halo_cells is None:
            halo_cells = cls.HALO_CELLS
        x = np.asarray(interior._x)
        y = np.asarray(interior._y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        x_halo = np.concatenate(
            (
                x[0] + dx * np.arange(-halo_cells, 0),
                x,
                x[-1] + dx * np.arange(1, halo_cells + 1),
            )
        )
        y_halo = np.concatenate(
            (
                y[0] + dy * np.arange(-halo_cells, 0),
                y,
                y[-1] + dy * np.arange(1, halo_cells + 1),
            )
        )
        ocean_vars = list(cls.FEATURE_NAMES["state"])
        return cls(
            n_timesteps=interior._n_timesteps,
            surface_band_width=cls.BOUNDARY_SURFACE_BAND_WIDTH,
            depth_band_width=cls.BOUNDARY_DEPTH_BAND_WIDTH,
            x_coords=x_halo,
            y_coords=y_halo,
            feature_names={
                "state": ocean_vars,
                "forcing": ocean_vars,
                "static": list(cls.FEATURE_NAMES["static"]),
            },
        )

    @property
    def root_path(self) -> Path:
        """
        The root path to the datastore. It is relative to this that any derived
        files (for example the graph components) are stored.

        Returns
        -------
        pathlib.Path
            The root path to the datastore.

        """
        return self._root_path

    @property
    def config(self) -> dict:
        """The configuration of the datastore.

        Returns
        -------
        collections.abc.Mapping
            The configuration of the datastore, any dict like object can be
            returned.

        """
        return {}

    @property
    def step_length(self) -> int:
        """The step length of the dataset in hours.

        Returns:
            int: The step length in hours.

        """
        return 1

    def get_vars_names(self, category: str) -> list[str]:
        """Get the names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        List[str]
            The names of the variables.

        """
        return self.ds[f"{category}_feature"].values.tolist()

    def get_vars_units(self, category: str) -> list[str]:
        """Get the units of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        List[str]
            The units of the variables.

        """
        return self.ds[f"{category}_feature_units"].values.tolist()

    def get_vars_long_names(self, category: str) -> List[str]:
        """Get the long names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        List[str]
            The long names of the variables.

        """
        return self.ds[f"{category}_feature_long_name"].values.tolist()

    def get_num_data_vars(self, category: str) -> int:
        """Get the number of data variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        int
            The number of data variables.

        """
        return self.ds[f"{category}_feature"].size

    def get_standardization_dataarray(self, category: str) -> xr.Dataset:
        """
        Return the standardization (i.e. scaling to mean of 0.0 and standard
        deviation of 1.0) dataarray for the given category. This should contain
        a `{category}_mean` and `{category}_std` variable for each variable in
        the category.
        For `category=="state"`, the dataarray should also contain a
        `state_diff_mean_standardized` and `state_diff_std_standardized`
        variable for the one-step differences of the state variables.
        The returned dataarray should at least have dimensions of
        `({category}_feature)`, but can also include for example `grid_index`
        (if the standardization is done per grid point for example).

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        xr.Dataset
            The standardization dataarray for the given category, with variables
            for the mean and standard deviation of the variables (and
            differences for state variables).

        """
        ds_standardization = xr.Dataset()

        ops = ["mean", "std"]
        if category == "state":
            ops += ["diff_mean_standardized", "diff_std_standardized"]

        for op in ops:
            da_op = xr.ones_like(self.ds[f"{category}_feature"]).astype(float)
            ds_standardization[f"{category}_{op}"] = da_op

        return ds_standardization

    def get_dataarray(
        self, category: str, split: str, standardize: bool = False
    ) -> Union[xr.DataArray, None]:
        """
        Return the processed data (as a single `xr.DataArray`) for the given
        category of data and test/train/val-split that covers all the data (in
        space and time) of a given category (state/forcing/static). A
        datastore must be able to return for the "state" category, but
        "forcing" and "static" are optional (in which case the method should
        return `None`). For the "static" category the `split` is allowed to be
        `None` because the static data is the same for all splits.

        The returned dataarray is expected to at minimum have dimensions of
        `(grid_index, {category}_feature)` so that any spatial dimensions have
        been stacked into a single dimension and all variables and levels have
        been stacked into a single feature dimension named by the `category` of
        data being loaded.

        For categories of data that have a time dimension (i.e. not static
        data), the dataarray is expected additionally have `(init_time,
        lead_time)` dimensions if `is_forecast` is True, or `(time)` if
        `is_forecast` is False.

        If the data is ensemble data, the dataarray is expected to have an
        additional `ensemble_member` dimension.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        split : str
            The time split to filter the dataset (train/val/test).
        standardize: bool
            If the dataarray should be returned standardized

        Returns
        -------
        xr.DataArray or None
            The xarray DataArray object with processed dataset.

        """
        dim_order = self.expected_dim_order(category=category)

        da_category = self.ds[category].transpose(*dim_order)

        if standardize:
            return self._standardize_datarray(da_category, category=category)

        return da_category

    @cached_property
    def boundary_mask(self) -> xr.DataArray:
        """
        Return the boundary mask for the dataset, with spatial dimensions
        stacked. Where the value is 1, the grid point is a boundary point, and
        where the value is 0, the grid point is not a boundary point.

        Returns
        -------
        xr.DataArray
            The boundary mask for the dataset, with dimensions
            `('grid_index',)`.

        """
        return self.ds["boundary_mask"]

    def get_xy(self, category: str, stacked: bool) -> ndarray:
        """Return the x, y coordinates of the dataset.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        stacked : bool
            Whether to stack the x, y coordinates.

        Returns
        -------
        np.ndarray
            The x, y coordinates of the dataset, returned differently based on
            the value of `stacked`:
            - `stacked==True`: shape `(n_grid_points, 2)` where
                               n_grid_points=N_x*N_y.
            - `stacked==False`: shape `(N_x, N_y, 2)`

        """
        # assume variables are stored in dimensions [grid_index, ...]
        ds_category = self.unstack_grid_coords(da_or_ds=self.ds[category])

        da_xs = ds_category.x
        da_ys = ds_category.y

        assert da_xs.ndim == da_ys.ndim == 1, "x and y coordinates must be 1D"

        da_x, da_y = xr.broadcast(da_xs, da_ys)
        da_xy = xr.concat([da_x, da_y], dim="grid_coord")

        if stacked:
            da_xy = da_xy.stack(grid_index=self.CARTESIAN_COORDS).transpose(
                "grid_index",
                "grid_coord",
            )
        else:
            dims = [
                "x",
                "y",
                "grid_coord",
            ]
            da_xy = da_xy.transpose(*dims)

        return da_xy.values

    @property
    def coords_projection(self) -> ccrs.Projection:
        """Return the projection object for the coordinates.

        The projection object is used to plot the coordinates on a map.

        Returns
        -------
        cartopy.crs.Projection:
            The projection object.

        """
        return ccrs.LambertConformal(
            central_longitude=20.0,
            central_latitude=60.0,
            standard_parallels=(60.0, 60.0),
            globe=ccrs.Globe(
                semimajor_axis=6371229.0, semiminor_axis=6371229.0
            ),
        )

    @property
    def num_grid_points(self) -> int:
        """Return the number of grid points in the dataset.

        Returns
        -------
        int
            The number of grid points in the dataset.

        """
        return self._num_grid_points

    @cached_property
    def grid_shape_state(self) -> CartesianGridShape:
        """The shape of the grid for the state variables.

        Returns
        -------
        CartesianGridShape:
            The shape of the grid for the state variables, which has `x` and
            `y` attributes.
        """

        n_points_1d = int(np.sqrt(self.num_grid_points))
        return CartesianGridShape(x=n_points_1d, y=n_points_1d)

    def get_projected_xy(self, category: str, stacked: bool) -> ndarray:
        """Dummy coordinates are already in the projected CRS."""
        return self.get_xy(category=category, stacked=stacked)

    @staticmethod
    def _horizontal_band(n: int, width: int) -> ndarray:
        """Centered band of ``width`` cells along y, spanning all x."""
        mask = np.zeros((n, n), dtype=bool)
        start = (n - width) // 2
        mask[:, start : start + width] = True
        return mask

    def get_mask(self, surface: bool, stacked: bool, invert: bool) -> ndarray:
        """
        Nested horizontal sea bands: wider at the surface, narrower at 50 m.

        Parameters
        ----------
        surface : bool
            If True, return only the surface (widest) band.
        stacked : bool
            If True, flatten ``x``/``y`` into ``grid_index``.
        invert : bool
            If True, return land / below-bathymetry (True = dry).
        """
        n_feat = self.get_num_data_vars("state")
        names = self.get_vars_names("state")
        mask = np.empty(
            (self.grid_shape_state.x, self.grid_shape_state.y, n_feat),
            dtype=bool,
        )
        for i, name in enumerate(names):
            mask[:, :, i] = (
                self._depth_band
                if name in self.DEEP_STATE_FEATURES
                else self._surface_band
            )
        if surface:
            mask_out = mask[:, :, 0]
        else:
            mask_out = mask
        if stacked:
            if surface:
                mask_out = mask_out.reshape(self.num_grid_points)
            else:
                mask_out = mask_out.reshape(self.num_grid_points, n_feat)
        if invert:
            mask_out = ~mask_out
        return mask_out

    def get_atmosphere_mask(
        self, stacked: bool = True, invert: bool = False
    ) -> ndarray:
        return self.get_mask(surface=True, stacked=stacked, invert=invert)
