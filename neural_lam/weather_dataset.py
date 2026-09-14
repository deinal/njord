# Standard library
import datetime
import warnings
from typing import Union

# Third-party
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
from neural_lam.datastore.base import BaseDatastore


class WeatherDataset(torch.utils.data.Dataset):
    """Dataset class for weather data.

    This class loads and processes weather data from a given datastore.

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore to load the data from.
    datastore_boundary : BaseDatastore
        The boundary datastore to load the data from.
    datastore_atmosphere : BaseDatastore
        The atmosphere datastore to load the data from.
    split : str, optional
        The data split to use ("train", "val" or "test"). Default is "train".
    ar_steps : int, optional
        The number of autoregressive steps. Default is 3.
    input_steps : int, optional
        Number of initial state steps (1 or 2). Default is 1.
    num_past_forcing_steps: int, optional
        Number of past time steps to include in forcing input. If set to i,
        forcing from times t-i, t-i+1, ..., t-1, t (and potentially beyond,
        given num_future_forcing_steps) are included as forcing inputs at time t
        Default is 1.
    num_future_forcing_steps: int, optional
        Number of future time steps to include in forcing input. If set to j,
        forcing from times t, t+1, ..., t+j-1, t+j (and potentially times before
        t, given num_past_forcing_steps) are included as forcing inputs at time
        t. Default is 1.
    current_forcing_step: bool, optional
        If True, the forcing window includes time t (current step).
        If False, only past (t-i..t-1) and future (t+1..t+j) are included, so
        you can use e.g. only future step(s) with num_past_forcing_steps=0.
        Default is True.
    current_boundary_step: bool, optional
        If True, boundary window includes time t. Default True.
    current_atmosphere_step: bool, optional
        If True, atmosphere window includes time t. Default True.
    num_past_boundary_steps: int, optional
        Number of past time steps to include in boundary input. If set to i,
        boundary from times t-i, t-i+1, ..., t-1, t (and potentially beyond,
        given num_future_boundary_steps) are included as boundary inputs at time
        t Default is 1.
    num_future_boundary_steps: int, optional
        Number of future time steps to include in boundary input. If set to j,
        boundary from times t, t+1, ..., t+j-1, t+j (and potentially times
        before t, given num_past_boundary_steps) are included as boundary inputs
        at time t. Default is 1.
    standardize : bool, optional
        Whether to standardize the data. Default is True.
    density_channel : DensityChannel or None, optional
        If provided, an binary density channel is constructed from
        the reference variable and appended to the state features.
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        datastore_boundary: BaseDatastore,
        datastore_atmosphere: BaseDatastore,
        split="train",
        ar_steps=1,
        input_steps=1,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        current_forcing_step=True,
        current_boundary_step=True,
        current_atmosphere_step=True,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
        num_past_atmosphere_steps=1,
        num_future_atmosphere_steps=1,
        standardize=True,
        use_atmosphere_g2m=False,
        density_channel=None,
        statistics_datastore: BaseDatastore = None,
        statistics_datastore_boundary: BaseDatastore = None,
        statistics_datastore_atmosphere: BaseDatastore = None,
    ):
        super().__init__()

        self.split = split
        self.ar_steps = ar_steps
        self.input_steps = input_steps
        self.datastore = datastore
        self.datastore_boundary = datastore_boundary
        self.datastore_atmosphere = datastore_atmosphere
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.current_forcing_step = current_forcing_step
        self.current_boundary_step = current_boundary_step
        self.current_atmosphere_step = current_atmosphere_step
        self.num_past_boundary_steps = num_past_boundary_steps
        self.num_future_boundary_steps = num_future_boundary_steps
        self.num_past_atmosphere_steps = num_past_atmosphere_steps
        self.num_future_atmosphere_steps = num_future_atmosphere_steps
        self.use_atmosphere_g2m = use_atmosphere_g2m
        self.density_channel = density_channel
        self.statistics_datastore = (
            statistics_datastore
            if statistics_datastore is not None
            else datastore
        )
        self.statistics_datastore_boundary = (
            statistics_datastore_boundary
            if statistics_datastore_boundary is not None
            else datastore_boundary
        )
        self.statistics_datastore_atmosphere = (
            statistics_datastore_atmosphere
            if statistics_datastore_atmosphere is not None
            else datastore_atmosphere
        )

        # Pre-compute index and standardized threshold for density channel
        if self.density_channel is not None:
            state_var_names = datastore.get_vars_names(category="state")
            ref_var = self.density_channel.reference_var
            if ref_var not in state_var_names:
                raise ValueError(
                    f"density_channel.reference_var='{ref_var}' "
                    f"not found in state variables: {state_var_names}"
                )
            self._ref_idx = state_var_names.index(ref_var)

        self.da_state = self.datastore.get_dataarray(
            category="state", split=self.split
        )
        if self.da_state is None:
            raise ValueError("A non-empty state dataarray must be provided.")
        self.da_forcing = self.datastore.get_dataarray(
            category="forcing", split=self.split
        )
        if self.datastore_boundary is not None:
            self.da_boundary_forcing = self.datastore_boundary.get_dataarray(
                category="forcing", split=self.split
            )
        else:
            self.da_boundary_forcing = None
        if self.datastore_atmosphere is not None:
            self.da_atmosphere_forcing = (
                self.datastore_atmosphere.get_dataarray(
                    category="forcing", split=self.split
                )
            )
        else:
            self.da_atmosphere_forcing = None
        self.surface_mask = self.datastore.get_mask(
            surface=True, stacked=True, invert=False
        )  # (N_lon*N_lat)
        self.land_mask = self.datastore.get_mask(
            surface=False, stacked=True, invert=True
        )  # (N_lon*N_lat, d_features)
        self.land_mask_bool = self.datastore.get_mask(
            surface=False, stacked=True, invert=True
        )[self.surface_mask][np.newaxis, ...].astype(
            bool
        )  # (1, N_grid, d_features)
        if self.datastore_boundary is not None:
            self.surface_mask_boundary = self.datastore_boundary.get_mask(
                surface=True, stacked=True, invert=False
            )  # (N_lon_boundary*N_lat_boundary)
            land_mask_boundary = self.datastore_boundary.get_mask(
                surface=False, stacked=True, invert=True
            )  # (N_lon_boundary*N_lat_boundary, d_features)
            self.land_mask_bool_boundary = land_mask_boundary[
                self.surface_mask_boundary
            ][np.newaxis, ...].astype(
                bool
            )  # (1, N_boundary_grid, d_features)
        else:
            self.surface_mask_boundary = None
            self.land_mask_bool_boundary = None
        if self.use_atmosphere_g2m:
            self.mask_atmosphere = (
                self.datastore_atmosphere.get_atmosphere_mask(
                    stacked=True, invert=False
                )
            )  # (N_lon_atmosphere*N_lat_atmosphere)
        else:
            self.mask_atmosphere = None

        # check that with the provided data-arrays and ar_steps that we have a
        # non-zero amount of samples
        if self.__len__() <= 0:
            raise ValueError(
                "The provided datastore only provides "
                f"{len(self.da_state.time)} total time steps, which is too few "
                "to create a single sample for the WeatherDataset "
                f"configuration used in the `{split}` split. You could try "
                "either reducing the number of autoregressive steps "
                "(`ar_steps`) and/or the forcing window size "
                "(`num_past_forcing_steps` and `num_future_forcing_steps`)"
            )

        # Check the dimensions and their ordering
        parts = dict(state=self.da_state)
        if self.da_forcing is not None:
            parts["forcing"] = self.da_forcing

        for part, da in parts.items():
            expected_dim_order = self.datastore.expected_dim_order(
                category=part
            )
            if da.dims != expected_dim_order:
                raise ValueError(
                    f"The dimension order of the `{part}` data ({da.dims}) "
                    f"does not match the expected dimension order "
                    f"({expected_dim_order}). Maybe you forgot to transpose "
                    "the data in `BaseDatastore.get_dataarray`?"
                )

        # Set up for standardization
        # TODO: This will become part of ar_model.py soon!
        self.standardize = standardize
        if standardize:
            self.ds_state_stats = (
                self.statistics_datastore.get_standardization_dataarray(
                    category="state"
                )
            )

            self.da_state_mean = self.ds_state_stats.state_mean
            self.da_state_std = self.ds_state_stats.state_std

            if self.density_channel is not None:
                mean_ref = float(
                    self.da_state_mean.isel(state_feature=self._ref_idx).values
                )
                std_ref = float(
                    self.da_state_std.isel(state_feature=self._ref_idx).values
                )
                self._ref_std_threshold = -mean_ref / std_ref

            if self.da_forcing is not None:
                self.ds_forcing_stats = (
                    self.statistics_datastore.get_standardization_dataarray(
                        category="forcing"
                    )
                )
                self.da_forcing_mean = self.ds_forcing_stats.forcing_mean
                self.da_forcing_std = self.ds_forcing_stats.forcing_std

            if self.da_boundary_forcing is not None:
                sdb = self.statistics_datastore_boundary
                self.ds_boundary_stats = sdb.get_standardization_dataarray(
                    category="forcing"
                )
                self.da_boundary_mean = self.ds_boundary_stats.forcing_mean
                self.da_boundary_std = self.ds_boundary_stats.forcing_std

            if self.da_atmosphere_forcing is not None:
                sda = self.statistics_datastore_atmosphere
                self.ds_atmosphere_stats = sda.get_standardization_dataarray(
                    category="forcing"
                )
                self.da_atmosphere_mean = self.ds_atmosphere_stats.forcing_mean
                self.da_atmosphere_std = self.ds_atmosphere_stats.forcing_std

    def __len__(self):
        if self.datastore.is_forecast:
            # for now we simply create a single sample for each analysis time
            # and then take the first (2 + ar_steps) forecast times. In
            # addition we only use the first ensemble member (if ensemble data
            # has been provided).
            # This means that for each analysis time we get a single sample

            if self.datastore.is_ensemble:
                warnings.warn(
                    "only using first ensemble member, so dataset size is "
                    " effectively reduced by the number of ensemble members "
                    f"({self.da_state.ensemble_member.size})",
                    UserWarning,
                )

            # check that there are enough forecast steps available to create
            # samples given the number of autoregressive steps requested
            n_forecast_steps = self.da_state.lead_time.size
            if n_forecast_steps < self.input_steps + self.ar_steps:
                raise ValueError(
                    "The number of forecast steps available "
                    f"({n_forecast_steps}) is less than the required "
                    f"input_steps+ar_steps "
                    f"({self.input_steps}+{self.ar_steps}) for "
                    "creating a sample with initial and target states."
                )

            return self.da_state.init_time.size
        else:
            # Calculate the number of samples in the dataset n_samples = total
            # time steps - (autoregressive steps + past forcing + future
            # forcing)
            #:
            # Where:
            #   - total time steps: len(self.da_state.time)
            #   - autoregressive steps: self.ar_steps
            #   - past forcing: max(input_steps, self.num_past_forcing_steps)
            #   - future forcing: self.num_future_forcing_steps
            return (
                len(self.da_state.time)
                - self.ar_steps
                - max(self.input_steps, self.num_past_forcing_steps)
                - self.num_future_forcing_steps
            )

    def _slice_state_time(self, da_state, idx, n_steps: int):
        """
        Produce a time slice of the given dataarray `da_state` (state) starting
        at `idx` and with `n_steps` steps. An `offset`is calculated based on the
        `num_past_forcing_steps` class attribute. `Offset` is used to offset the
        start of the sample, to assert that enough previous time steps are
        available for the 2 initial states and any corresponding forcings
        (calculated in `_slice_forcing_time`).

        Parameters
        ----------
        da_state : xr.DataArray
            The dataarray to slice. This is expected to have a `time` dimension
            if the datastore is providing analysis only data, and
            `init_time` and `lead_time` dimensions if the datastore is
            providing forecast data.
        idx : int
            The index of the time step to start the sample from.
        n_steps : int
            The number of time steps to include in the sample.

        Returns
        -------
        da_sliced : xr.DataArray
            The sliced dataarray with dims ('time', 'grid_index',
            'state_feature').
        """
        init_steps = self.input_steps
        # slice the dataarray to include the required number of time steps
        if self.datastore.is_forecast:
            start_idx = max(0, self.num_past_forcing_steps - init_steps)
            end_idx = max(init_steps, self.num_past_forcing_steps) + n_steps
            # This implies that the data will have both `init_time` and
            # `lead_time` dimensions for forecasts. We select an init time
            # and the corresponding lead time range. One sample per forecast,
            # always starting from the earliest available lead time.
            da_sliced = da_state.isel(
                init_time=idx,
                lead_time=slice(start_idx, end_idx),
            )
            # create a new time dimension so that the produced sample has a
            # `time` dimension, similarly to the analysis only data
            da_sliced["time"] = da_sliced.init_time + da_sliced.lead_time
            da_sliced = da_sliced.swap_dims({"lead_time": "time"})
        else:
            # For analysis data we slice the time dimension directly. The offset
            # is only relevant for the very first (and last) samples in the
            # dataset.
            start_idx = idx + max(0, self.num_past_forcing_steps - init_steps)
            end_idx = (
                idx + max(init_steps, self.num_past_forcing_steps) + n_steps
            )
            da_sliced = da_state.isel(time=slice(start_idx, end_idx))
        return da_sliced

    def _slice_forcing_time(
        self,
        da_forcing,
        idx,
        n_steps: int,
        num_past_steps=None,
        num_future_steps=None,
        include_current=None,
    ):
        """
        Produce a time slice of the given dataarray `da_forcing` (forcing)
        starting at `idx` and with `n_steps` steps. An `offset` is calculated
        based on the `num_past_steps` parameter. It is used to offset the
        start of the sample, to ensure that enough previous time steps
        are available for the forcing data. The forcing data is windowed
        around the current autoregressive time step to include the past and
        future forcings.

        Parameters
        ----------
        da_forcing : xr.DataArray
            The forcing dataarray to slice. This is expected to have a `time`
            dimension if the datastore is providing analysis only data, and
            `init_time` and `lead_time` dimensions if the datastore is
            providing forecast data.
        idx : int
            The index of the time step to start the sample from.
        n_steps : int
            The number of time steps to include in the sample.
        num_past_steps : int, optional
            Number of past time steps to include in the window. If None, uses
            `self.num_past_forcing_steps`.
        num_future_steps : int, optional
            Number of future time steps to include in the window. If None, uses
            `self.num_future_forcing_steps`.
        include_current : bool, optional
            If None, uses self.current_forcing_step (grid). If given,
            use for boundary/atmosphere slices.

        Returns
        -------
        da_concat : xr.DataArray
            The sliced dataarray with dims ('time', 'grid_index',
            'window', 'forcing_feature').
        """
        # Use default forcing steps if not provided
        if num_past_steps is None:
            num_past_steps = self.num_past_forcing_steps
        if num_future_steps is None:
            num_future_steps = self.num_future_forcing_steps
        if include_current is None:
            include_current = getattr(self, "current_forcing_step", True)

        # The current implementation requires at least 2 time steps for the
        # initial state (see GraphCast). The forcing data is windowed around the
        # current autregressive time step. The two `init_steps` can also be used
        # as past forcings.
        init_steps = 2
        da_list = []

        if self.datastore.is_forecast:
            # This implies that the data will have both `init_time` and
            # `lead_time` dimensions for forecasts. We select an init time
            # and the corresponding lead time range. One sample per forecast.
            # Add a 'time' dimension using the actual forecast times.
            offset = max(init_steps, num_past_steps)
            for step in range(n_steps):
                start_idx = offset + step - num_past_steps
                end_idx = offset + step + num_future_steps
                current_idx = offset + step

                current_time = (
                    da_forcing.init_time[idx]
                    + da_forcing.lead_time[current_idx]
                )

                if include_current:
                    da_sliced = da_forcing.isel(
                        init_time=idx,
                        lead_time=slice(start_idx, end_idx + 1),
                    )
                else:
                    parts = []
                    if num_past_steps > 0:
                        da_past = da_forcing.isel(
                            init_time=idx,
                            lead_time=slice(start_idx, current_idx),
                        )
                        parts.append(da_past)
                    if num_future_steps > 0:
                        da_future = da_forcing.isel(
                            init_time=idx,
                            lead_time=slice(current_idx + 1, end_idx + 1),
                        )
                        parts.append(da_future)
                    da_sliced = (
                        xr.concat(parts, dim="lead_time")
                        if len(parts) > 1
                        else parts[0]
                    )

                da_sliced = da_sliced.rename({"lead_time": "window"})

                # Assign the 'window' coordinate to be relative positions
                da_sliced = da_sliced.assign_coords(
                    window=np.arange(len(da_sliced.window))
                )

                if "time" in da_sliced.coords:
                    # time may exist if used to calculate sine/cosine fts
                    da_sliced = da_sliced.drop_vars("time")
                da_sliced = da_sliced.expand_dims(
                    dim={"time": [current_time.values]}
                )

                da_list.append(da_sliced)

            # Concatenate the list of DataArrays along the 'time' dimension
            da_concat = xr.concat(da_list, dim="time")

        else:
            # For analysis data, we slice the time dimension directly. The
            # offset is only relevant for the very first (and last) samples in
            # the dataset.
            offset = idx + max(init_steps, num_past_steps)
            for step in range(n_steps):
                start_idx = offset + step - num_past_steps
                end_idx = offset + step + num_future_steps
                current_idx = offset + step

                if include_current:
                    da_sliced = da_forcing.isel(
                        time=slice(start_idx, end_idx + 1)
                    )
                else:
                    parts = []
                    if num_past_steps > 0:
                        da_past = da_forcing.isel(
                            time=slice(start_idx, current_idx)
                        )
                        parts.append(da_past)
                    if num_future_steps > 0:
                        da_future = da_forcing.isel(
                            time=slice(current_idx + 1, end_idx + 1)
                        )
                        parts.append(da_future)
                    da_sliced = (
                        xr.concat(parts, dim="time")
                        if len(parts) > 1
                        else parts[0]
                    )

                da_sliced = da_sliced.rename({"time": "window"})

                # Assign the 'window' coordinate to be relative positions
                da_sliced = da_sliced.assign_coords(
                    window=np.arange(len(da_sliced.window))
                )

                # Add a 'time' dimension to keep track of steps using actual
                # time coordinates
                current_time = da_forcing.time[current_idx]
                da_sliced = da_sliced.expand_dims(
                    dim={"time": [current_time.values]}
                )

                da_list.append(da_sliced)

            # Concatenate the list of DataArrays along the 'time' dimension
            da_concat = xr.concat(da_list, dim="time")

        return da_concat

    def _process_windowed_data(
        self,
        da_windowed,
        da_state,
        da_target_times,
    ):
        """Helper function to process windowed data. This function stacks the
        'forcing_feature' and 'window' dimensions.

        Parameters
        ----------
        da_windowed : xr.DataArray
            The windowed data to process. Can be `None` if no data is provided.
        da_state : xr.DataArray
            The state dataarray.
        da_target_times : xr.DataArray
            The target times.

        Returns
        -------
        da_windowed : xr.DataArray
            The processed windowed data. If `da_windowed` is `None`, an empty
            DataArray with the correct dimensions and coordinates is returned.

        """
        stacked_dim = "forcing_feature_windowed"
        if da_windowed is not None:
            # Stack the 'feature' and 'window' dimensions
            da_windowed = da_windowed.stack(
                {stacked_dim: ("forcing_feature", "window")}
            )
        else:
            # Create empty DataArray with the correct dimensions and coordinates
            da_windowed = xr.DataArray(
                data=np.empty((self.ar_steps, da_state.grid_index.size, 0)),
                dims=("time", "grid_index", f"{stacked_dim}"),
                coords={
                    "time": da_target_times,
                    "grid_index": da_state.grid_index,
                    f"{stacked_dim}": [],
                },
            )
        return da_windowed

    def _build_item_dataarrays(self, idx):
        """
        Create the dataarrays for the initial states, target states, forcing,
        boundary and atmosphere data for the sample at index `idx`.

        Parameters
        ----------
        idx : int
            The index of the sample to create the dataarrays for.

        Returns
        -------
        da_init_states : xr.DataArray
            The dataarray for the initial states.
        da_target_states : xr.DataArray
            The dataarray for the target states.
        da_forcing_windowed : xr.DataArray
            The dataarray for the forcing data, windowed for the sample.
        da_boundary_windowed : xr.DataArray
            The dataarray for the boundary data, windowed for the sample.
        da_atmosphere_windowed : xr.DataArray
            The dataarray for the atmosphere data, windowed for the sample.
        da_target_times : xr.DataArray
            The dataarray for the target times.
        """
        # handling ensemble data
        if self.datastore.is_ensemble:
            # for the now the strategy is to only include the first ensemble
            # member
            # XXX: this could be changed to include all ensemble members by
            # splitting `idx` into two parts, one for the analysis time and one
            # for the ensemble member and then increasing self.__len__ to
            # include all ensemble members
            warnings.warn(
                "only use of ensemble member 0 (the first member) is "
                "implemented for ensemble data"
            )
            i_ensemble = 0
            da_state = self.da_state.isel(ensemble_member=i_ensemble)
        else:
            da_state = self.da_state

        if self.da_forcing is not None:
            if "ensemble_member" in self.da_forcing.dims:
                raise NotImplementedError(
                    "Ensemble member not yet supported for forcing data"
                )
            da_forcing = self.da_forcing
        else:
            da_forcing = None

        if self.da_boundary_forcing is not None:
            da_boundary = self.da_boundary_forcing
        else:
            da_boundary = None

        if self.da_atmosphere_forcing is not None:
            da_atmosphere = self.da_atmosphere_forcing
        else:
            da_atmosphere = None

        # handle time sampling in a way that is compatible with both analysis
        # and forecast data
        da_state = self._slice_state_time(
            da_state=da_state, idx=idx, n_steps=self.ar_steps
        )
        if da_forcing is not None:
            da_forcing_windowed = self._slice_forcing_time(
                da_forcing=da_forcing, idx=idx, n_steps=self.ar_steps
            )
        else:
            da_forcing_windowed = None
        if da_boundary is not None:
            da_boundary_windowed = self._slice_forcing_time(
                da_forcing=da_boundary,
                idx=idx,
                n_steps=self.ar_steps,
                num_past_steps=self.num_past_boundary_steps,
                num_future_steps=self.num_future_boundary_steps,
                include_current=self.current_boundary_step,
            )
        else:
            da_boundary_windowed = None
        if da_atmosphere is not None:
            da_atmosphere_windowed = self._slice_forcing_time(
                da_forcing=da_atmosphere,
                idx=idx,
                n_steps=self.ar_steps,
                num_past_steps=self.num_past_atmosphere_steps,
                num_future_steps=self.num_future_atmosphere_steps,
                include_current=self.current_atmosphere_step,
            )
        else:
            da_atmosphere_windowed = None

        # load the data into memory
        da_state.load()
        if da_forcing is not None:
            da_forcing_windowed.load()
        if da_boundary is not None:
            da_boundary_windowed.load()
        if da_atmosphere is not None:
            da_atmosphere_windowed.load()

        da_init_states = da_state.isel(time=slice(0, self.input_steps))
        da_target_states = da_state.isel(time=slice(self.input_steps, None))
        da_target_times = da_target_states.time

        if self.standardize:
            da_init_states = (
                da_init_states - self.da_state_mean
            ) / self.da_state_std
            da_target_states = (
                da_target_states - self.da_state_mean
            ) / self.da_state_std

            if da_forcing is not None:
                # XXX: Here we implicitly assume that the last dimension of the
                # forcing data is the forcing feature dimension. To standardize
                # on `.device` we need a different implementation. (e.g. a
                # tensor with repeated means and stds for each "windowed" time.)
                da_forcing_windowed = (
                    da_forcing_windowed - self.da_forcing_mean
                ) / self.da_forcing_std

            if da_boundary is not None:
                da_boundary_windowed = (
                    da_boundary_windowed - self.da_boundary_mean
                ) / self.da_boundary_std

            if da_atmosphere is not None:
                da_atmosphere_windowed = (
                    da_atmosphere_windowed - self.da_atmosphere_mean
                ) / self.da_atmosphere_std

        # stack the `forcing_feature` and `window` dimensions
        da_forcing_windowed = self._process_windowed_data(
            da_forcing_windowed,
            da_state,
            da_target_times,
        )
        da_boundary_windowed = self._process_windowed_data(
            da_boundary_windowed,
            da_state,
            da_target_times,
        )
        da_atmosphere_windowed = self._process_windowed_data(
            da_atmosphere_windowed,
            da_state,
            da_target_times,
        )

        return (
            da_init_states,
            da_target_states,
            da_forcing_windowed,
            da_boundary_windowed,
            da_atmosphere_windowed,
            da_target_times,
        )

    def __getitem__(self, idx):
        """
        Return a single training sample, which consists of the initial states,
        target states, forcing and batch times.

        The implementation currently uses xarray.DataArray objects for the
        standardization (scaling to mean 0.0 and standard deviation of 1.0) so
        that we can make us of xarray's broadcasting capabilities. This makes
        it possible to standardization with both global means, but also for
        example where a grid-point mean has been computed. This code will have
        to be replace if standardization is to be done on the GPU to handle
        different shapes of the standardization.

        Parameters
        ----------
        idx : int
            The index of the sample to return, this will refer to the time of
            the initial state.

        Returns
        -------
        init_states : TrainingSample
            A training sample object containing the initial states, target
            states, forcing and batch times. The batch times are the times of
            the target steps.

        """
        (
            da_init_states,
            da_target_states,
            da_forcing_windowed,
            da_boundary_windowed,
            da_atmosphere_windowed,
            da_target_times,
        ) = self._build_item_dataarrays(idx=idx)

        tensor_dtype = torch.float32

        init_states = torch.tensor(da_init_states.values, dtype=tensor_dtype)
        target_states = torch.tensor(
            da_target_states.values, dtype=tensor_dtype
        )

        target_times = torch.tensor(
            da_target_times.astype("datetime64[ns]").astype("int64").values,
            dtype=torch.int64,
        )

        forcing = torch.tensor(da_forcing_windowed.values, dtype=tensor_dtype)
        boundary = torch.tensor(da_boundary_windowed.values, dtype=tensor_dtype)
        atmosphere = torch.tensor(
            da_atmosphere_windowed.values, dtype=tensor_dtype
        )

        # mask to surface grid (N_lon*N_lat -> N_grid)
        init_states = init_states[:, self.surface_mask, :]
        target_states = target_states[:, self.surface_mask, :]
        forcing = forcing[:, self.surface_mask, :]
        if self.surface_mask_boundary is not None:
            boundary = boundary[:, self.surface_mask_boundary, :]
        if self.use_atmosphere_g2m:
            atmosphere = atmosphere[:, self.mask_atmosphere, :]
        else:
            # Same grid as interior (exact same grid size)
            atmosphere = atmosphere[:, self.surface_mask, :]

        # convert land from nan to zero
        init_states = torch.where(
            torch.tensor(self.land_mask_bool, dtype=torch.bool),
            torch.tensor(0.0, dtype=tensor_dtype),
            init_states,
        )
        target_states = torch.where(
            torch.tensor(self.land_mask_bool, dtype=torch.bool),
            torch.tensor(0.0, dtype=tensor_dtype),
            target_states,
        )

        # convert boundary land regions to zero
        if self.datastore_boundary is not None:
            land_mask_bool_boundary_tensor = torch.tensor(
                self.land_mask_bool_boundary, dtype=torch.bool
            )  # (1, N_boundary_grid, d_features)
            # repeat mask num_windows times to match boundary window size
            # (same as _slice_forcing_time: past + future + current if included)
            num_boundary_windows = (
                self.num_past_boundary_steps
                + self.num_future_boundary_steps
                + (1 if self.current_boundary_step else 0)
            )
            # intereleaved repeat along feature dimension to match windowing
            # stacked as: (feature_0, window_0), (feature_0, window_1), ...,
            # (feature_1, window_0), (feature_1, window_1), ...
            land_mask_bool_boundary_tensor = (
                land_mask_bool_boundary_tensor.repeat_interleave(
                    num_boundary_windows, dim=2
                )
            )
            boundary = torch.where(
                land_mask_bool_boundary_tensor,
                torch.tensor(0.0, dtype=tensor_dtype),
                boundary,
            )

        # Construct and append ice density channel.
        # density = 1 where reference variable > 0 in physical space, else 0.
        # Land nodes get density = 0 via the land mask.
        if self.density_channel is not None:
            idx = self._ref_idx
            if self.standardize:
                thr = self._ref_std_threshold
            else:
                thr = 0.0
            land_ref = torch.tensor(
                self.land_mask_bool[0, :, idx], dtype=torch.bool
            )  # (N_grid,), True for land
            init_density = (init_states[:, :, idx] > thr).float()
            init_density[:, land_ref] = 0.0
            target_density = (target_states[:, :, idx] > thr).float()
            target_density[:, land_ref] = 0.0
            init_states = torch.cat(
                [init_states, init_density.unsqueeze(-1)], dim=-1
            )
            target_states = torch.cat(
                [target_states, target_density.unsqueeze(-1)], dim=-1
            )

        # init_states: (2, N_grid, d_features)
        # target_states: (ar_steps, N_grid, d_features)
        # forcing: (ar_steps, N_grid, d_windowed_forcing)
        # boundary: (ar_steps, N_boundary_grid, d_windowed_boundary)
        # atmosphere: (ar_steps, N_atmosphere_grid, d_windowed_atmosphere)
        # target_times: (ar_steps,)

        return (
            init_states,
            target_states,
            forcing,
            boundary,
            atmosphere,
            target_times,
        )

    def __iter__(self):
        """
        Convenience method to iterate over the dataset.

        This isn't used by pytorch DataLoader which itself implements an
        iterator that uses Dataset.__getitem__ and Dataset.__len__.

        """
        for i in range(len(self)):
            yield self[i]

    def create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: Union[datetime.datetime, list[datetime.datetime]],
        category: str,
    ):
        """
        Construct a xarray.DataArray from a `pytorch.Tensor` with coordinates
        for `grid_index`, `time` and `{category}_feature` matching the shape
        and number of times provided and add the x/y coordinates from the
        datastore.

        The number of times provided is expected to match the shape of the
        tensor. For a 2D tensor, the dimensions are assumed to be (grid_index,
        {category}_feature) and only a single time should be provided. For a 3D
        tensor, the dimensions are assumed to be (time, grid_index,
        {category}_feature) and a list of times should be provided.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to construct the DataArray from, this assumed to have
            the same dimension ordering as returned by the __getitem__ method
            (i.e. time, grid_index, {category}_feature). The tensor will be
            copied to the CPU before constructing the DataArray.
        time : datetime.datetime or list[datetime.datetime]
            The time or times of the tensor.
        category : str
            The category of the tensor, either "state", "forcing" or "static".

        Returns
        -------
        da : xr.DataArray
            The constructed DataArray.
        """

        def _is_listlike(obj):
            # match list, tuple, numpy array
            return hasattr(obj, "__iter__") and not isinstance(obj, str)

        add_time_as_dim = False
        if len(tensor.shape) == 2:
            dims = ["grid_index", f"{category}_feature"]
            if _is_listlike(time):
                raise ValueError(
                    "Expected a single time for a 2D tensor with assumed "
                    "dimensions (grid_index, {category}_feature), but got "
                    f"{len(time)} times"
                )
        elif len(tensor.shape) == 3:
            add_time_as_dim = True
            dims = ["time", "grid_index", f"{category}_feature"]
            if not _is_listlike(time):
                raise ValueError(
                    "Expected a list of times for a 3D tensor with assumed "
                    "dimensions (time, grid_index, {category}_feature), but "
                    "got a single time"
                )
        else:
            raise ValueError(
                "Expected tensor to have 2 or 3 dimensions, but got "
                f"{len(tensor.shape)}"
            )

        da_datastore_state = getattr(self, f"da_{category}")
        da_grid_index = da_datastore_state.grid_index
        da_state_feature = da_datastore_state.state_feature

        coords = {
            f"{category}_feature": da_state_feature,
            "grid_index": da_grid_index,
        }

        array = tensor.cpu().numpy()
        d_full_grid = self.surface_mask.shape[0]
        d_features = array.shape[-1]

        if add_time_as_dim:
            coords["time"] = time
            d_time = array.shape[0]
            full_array = np.zeros((d_time, d_full_grid, d_features))
            full_array[:, self.surface_mask, :] = array
            full_array = np.where(
                self.land_mask[np.newaxis, ...], np.nan, full_array
            )
        else:
            full_array = np.zeros((d_full_grid, d_features))
            full_array[self.surface_mask, :] = array
            full_array = np.where(self.land_mask, np.nan, full_array)

        da = xr.DataArray(
            full_array,
            dims=dims,
            coords=coords,
        )

        for grid_coord in ["longitude", "latitude"]:
            if (
                grid_coord in da_datastore_state.coords
                and grid_coord not in da.coords
            ):
                da.coords[grid_coord] = da_datastore_state[grid_coord]

        if not add_time_as_dim:
            da.coords["time"] = time

        return da


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        datastore: BaseDatastore,
        datastore_boundary: BaseDatastore,
        datastore_atmosphere: BaseDatastore,
        ar_steps_train=3,
        ar_steps_eval=25,
        input_steps=1,
        standardize=True,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        current_forcing_step=True,
        current_boundary_step=True,
        current_atmosphere_step=True,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
        num_past_atmosphere_steps=1,
        num_future_atmosphere_steps=1,
        batch_size=4,
        num_workers=16,
        use_atmosphere_g2m=False,
        density_channel=None,
        statistics_datastore: BaseDatastore = None,
        statistics_datastore_boundary: BaseDatastore = None,
        statistics_datastore_atmosphere: BaseDatastore = None,
    ):
        super().__init__()
        self._datastore = datastore
        self._datastore_boundary = datastore_boundary
        self._datastore_atmosphere = datastore_atmosphere
        self._statistics_datastore = statistics_datastore
        self._statistics_datastore_boundary = statistics_datastore_boundary
        self._statistics_datastore_atmosphere = statistics_datastore_atmosphere
        self.use_atmosphere_g2m = use_atmosphere_g2m
        self.density_channel = density_channel
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.current_forcing_step = current_forcing_step
        self.current_boundary_step = current_boundary_step
        self.current_atmosphere_step = current_atmosphere_step
        self.num_past_boundary_steps = num_past_boundary_steps
        self.num_future_boundary_steps = num_future_boundary_steps
        self.num_past_atmosphere_steps = num_past_atmosphere_steps
        self.num_future_atmosphere_steps = num_future_atmosphere_steps
        self.ar_steps_train = ar_steps_train
        self.ar_steps_eval = ar_steps_eval
        self.input_steps = input_steps
        self.standardize = standardize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        if num_workers > 0:
            # default to spawn for now, as the default on linux "fork" hangs
            # when using dask
            self.multiprocessing_context = "spawn"
        else:
            self.multiprocessing_context = None

    def setup(self, stage=None):
        stats_kwargs = dict(
            statistics_datastore=self._statistics_datastore,
            statistics_datastore_boundary=self._statistics_datastore_boundary,
            statistics_datastore_atmosphere=(
                self._statistics_datastore_atmosphere
            ),
        )
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                datastore=self._datastore,
                datastore_boundary=self._datastore_boundary,
                datastore_atmosphere=self._datastore_atmosphere,
                split="train",
                ar_steps=self.ar_steps_train,
                input_steps=self.input_steps,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
                current_forcing_step=self.current_forcing_step,
                current_boundary_step=self.current_boundary_step,
                current_atmosphere_step=self.current_atmosphere_step,
                num_past_boundary_steps=self.num_past_boundary_steps,
                num_future_boundary_steps=self.num_future_boundary_steps,
                num_past_atmosphere_steps=self.num_past_atmosphere_steps,
                num_future_atmosphere_steps=self.num_future_atmosphere_steps,
                use_atmosphere_g2m=self.use_atmosphere_g2m,
                density_channel=self.density_channel,
                **stats_kwargs,
            )
            self.val_dataset = WeatherDataset(
                datastore=self._datastore,
                datastore_boundary=self._datastore_boundary,
                datastore_atmosphere=self._datastore_atmosphere,
                split="val",
                ar_steps=self.ar_steps_eval,
                input_steps=self.input_steps,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
                current_forcing_step=self.current_forcing_step,
                current_boundary_step=self.current_boundary_step,
                current_atmosphere_step=self.current_atmosphere_step,
                num_past_boundary_steps=self.num_past_boundary_steps,
                num_future_boundary_steps=self.num_future_boundary_steps,
                num_past_atmosphere_steps=self.num_past_atmosphere_steps,
                num_future_atmosphere_steps=self.num_future_atmosphere_steps,
                use_atmosphere_g2m=self.use_atmosphere_g2m,
                density_channel=self.density_channel,
                **stats_kwargs,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                datastore=self._datastore,
                datastore_boundary=self._datastore_boundary,
                datastore_atmosphere=self._datastore_atmosphere,
                split="test",
                ar_steps=self.ar_steps_eval,
                input_steps=self.input_steps,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
                current_forcing_step=self.current_forcing_step,
                current_boundary_step=self.current_boundary_step,
                current_atmosphere_step=self.current_atmosphere_step,
                num_past_boundary_steps=self.num_past_boundary_steps,
                num_future_boundary_steps=self.num_future_boundary_steps,
                num_past_atmosphere_steps=self.num_past_atmosphere_steps,
                num_future_atmosphere_steps=self.num_future_atmosphere_steps,
                use_atmosphere_g2m=self.use_atmosphere_g2m,
                density_channel=self.density_channel,
                **stats_kwargs,
            )

    def train_dataloader(self):
        """Load train dataset."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Load validation dataset."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Load test dataset."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=self.num_workers > 0,
        )
