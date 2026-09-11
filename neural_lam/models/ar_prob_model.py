# Standard library
import os
from typing import Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import xarray as xr
from loguru import logger

# Local
from .. import metrics, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..datastore.base import BaseRegularGridDatastore
from .ar_model import ARModel


class ARProbModel(ARModel):
    """
    Generic auto-regressive probabilistic weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
        datastore_atmosphere: Union[BaseDatastore, None],
        **kwargs,
    ):
        super().__init__(
            args,
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
            datastore_atmosphere=datastore_atmosphere,
            **kwargs,
        )

        self.ensemble_size = args.ensemble_size

        # Per-rank RNG for reproducible but distinct noise across DDP ranks.
        self._rank = self._get_rank()
        self._rng_generators = {}  # device -> torch.Generator

        # Tracks whether the test-set ensemble zarr store has been
        # initialized (first batch written with mode="w"); subsequent
        # batches append along init_time.
        self._forecasts_zarr_initialized = False

        self.val_metrics.update(
            {
                "ens_mse": [],
                "spread_squared": [],
            }
        )

        self.test_metrics.update(
            {
                "ens_mae": [],
                "ens_mse": [],
                "crps_ens": [],
                "spread_squared": [],
            }
        )

    def _get_rank(self) -> int:
        """Current process rank (0 if not distributed) used for per-rank RNG."""
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return torch.distributed.get_rank()
        return int(os.environ.get("LOCAL_RANK", 0))

    def _get_generator(self, device: torch.device) -> torch.Generator:
        """
        Per-rank Generator for noise sampling. Seeded with args.seed + rank so
        each DDP rank has a different but reproducible stream.
        """
        if device not in self._rng_generators:
            self._rng_generators[device] = torch.Generator(
                device=device
            ).manual_seed(self.args.seed + self._rank)
        return self._rng_generators[device]

    def sample_trajectories(
        self,
        init_states: torch.Tensor,
        forcing: torch.Tensor,
        boundary_forcing: torch.Tensor,
        atmosphere_forcing: torch.Tensor,
        ensemble_size: int = None,
    ):
        """
        Sample trajectories from the model.
        init_states: (B, 2, num_interior_nodes, d_f)
        forcing: (B, num_interior_nodes, d_static_f)
        boundary_forcing: (B, num_boundary_nodes, d_boundary_f)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)

        Returns:
        sampled_trajectories: (num_traj, B, pred_steps, num_interior_nodes, d_f)
        """
        if ensemble_size is None:
            ensemble_size = self.ensemble_size

        traj_list = [
            self.unroll_prediction(
                init_states,
                forcing,
                boundary_forcing,
                atmosphere_forcing,
            )
            for _ in range(ensemble_size)
        ]

        traj_tensor = torch.stack(
            [pred_pair[0] for pred_pair in traj_list], dim=1
        )

        return traj_tensor

    def ensemble_step(self, batch):
        """
        Perform ensemble forecast and compute basic metrics.
        Common step done during both evaluation and testing

        batch: tuple of tensors, batch to perform ensemble forecast on

        Returns:
        trajectories: (B, S, pred_steps, num_grid_nodes, d_f)
        traj_stds: (B, S, pred_steps, num_grid_nodes, d_f)
        target_states: (B, pred_steps, num_grid_nodes, d_f)
        spread_squared_batch: (B, pred_steps, d_f)
        ens_mse_batch: (B, pred_steps, d_f)
        """
        # Compute and store metrics for ensemble forecast
        (
            init_states,
            target_states,
            forcing_features,
            boundary_forcing,
            atmosphere_forcing,
            _,
        ) = batch

        trajectories = self.sample_trajectories(
            init_states,
            forcing_features,
            boundary_forcing,
            atmosphere_forcing,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        spread_squared_batch = metrics.spread_squared(
            trajectories,
            target_states,
            None,
            mask=self.loss_mask,
            sum_vars=False,
        )
        # (B, pred_steps, d_f)

        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_mse_batch = metrics.mse(
            ens_mean,
            target_states,
            None,
            mask=self.loss_mask,
            sum_vars=False,
        )  # (B, pred_steps, d_f)

        return (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        )

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        self.log_spsk_ratio(self.val_metrics, "val")
        super().on_validation_epoch_end()

    def _save_ensemble_example_to_zarr(
        self,
        time_row: torch.Tensor,
        trajectories_s_t_n_d: torch.Tensor,
        target_t_n_d: torch.Tensor,
        example_index: int,
    ):
        """
        Save one example's full ensemble (all members) and target to zarr.

        Writes ``example_forecasts/ensemble_example_{example_index}.zarr`` under
        ``logger.save_dir``, next to deterministic ``example_*.zarr`` files.
        Variables: ``prediction`` (ensemble_member, ...), ``target`` (...).

        Uses ``init_time`` and ``lead_time`` (timedelta) as forecast
        coordinates, with 1-day step size.
        """
        save_dir = os.path.join(self.logger.save_dir, "example_forecasts")
        os.makedirs(save_dir, exist_ok=True)
        zarr_path = os.path.join(
            save_dir, f"ensemble_example_{example_index}.zarr"
        )

        member_das = []
        s_count = trajectories_s_t_n_d.shape[0]
        for s in range(s_count):
            scaled = trajectories_s_t_n_d[s] * self.state_std + self.state_mean
            da_m = self._create_dataarray_from_tensor(
                tensor=scaled,
                time=time_row,
                split="test",
                category="state",
            )
            if isinstance(self._datastore, BaseRegularGridDatastore):
                da_m = self._datastore.unstack_grid_coords(da_m)

            t0 = da_m.coords["time"].values[0] - np.array(
                self._datastore.step_length, dtype="timedelta64[h]"
            )
            da_m.coords["init_time"] = t0
            da_m.coords["lead_time"] = da_m.time - t0
            da_m = da_m.swap_dims({"time": "lead_time"})
            member_das.append(da_m)

        ens_dim = xr.DataArray(
            np.arange(s_count, dtype=np.int64),
            dims=("ensemble_member",),
            name="ensemble_member",
        )
        da_prediction = xr.concat(member_das, dim=ens_dim)
        da_prediction.name = "prediction"

        target_scaled = target_t_n_d * self.state_std + self.state_mean
        da_target = self._create_dataarray_from_tensor(
            tensor=target_scaled,
            time=time_row,
            split="test",
            category="state",
        )
        if isinstance(self._datastore, BaseRegularGridDatastore):
            da_target = self._datastore.unstack_grid_coords(da_target)
        t0_t = da_target.coords["time"].values[0] - np.array(
            self._datastore.step_length, dtype="timedelta64[h]"
        )
        da_target.coords["init_time"] = t0_t
        da_target.coords["lead_time"] = da_target.time - t0_t
        da_target = da_target.swap_dims({"time": "lead_time"})
        da_target.name = "target"

        ds_out = xr.Dataset({"prediction": da_prediction, "target": da_target})
        n_feat = ds_out.sizes["state_feature"]
        n_lon = ds_out.sizes["longitude"]
        n_lat = ds_out.sizes["latitude"]
        encoding = {
            "init_time": {
                "units": "Seconds since 1970-01-01 00:00:00",
                "dtype": "int64",
            },
            "prediction": {
                # one chunk per (ensemble_member, lead_time);
                # all features + full spatial
                "chunks": (1, 1, n_feat, n_lon, n_lat),
            },
            "target": {
                # one chunk per lead_time; all features + full spatial
                "chunks": (1, n_feat, n_lon, n_lat),
            },
        }

        logger.info(f"Saving ensemble example to {zarr_path}")
        ds_out.to_zarr(
            zarr_path,
            mode="w",
            consolidated=True,
            encoding=encoding,
        )

    def _save_ensemble_batch_to_zarr(
        self,
        time_rows: torch.Tensor,
        trajectories: torch.Tensor,
        target_states: torch.Tensor,
        zarr_path: str,
        first_write: bool,
    ) -> None:
        """
        Append one test batch's ensemble predictions and targets to a single
        zarr store along ``init_time``.

        Shapes:
            time_rows: (B, pred_steps)
            trajectories: (B, S, pred_steps, num_grid_nodes, d_f)
            target_states: (B, pred_steps, num_grid_nodes, d_f)

        The store is created on the first call (``first_write=True``) with
        mode="w" and then extended with mode="a", append_dim="init_time" on
        subsequent calls. Intended to be invoked from rank 0 only.
        """
        B = trajectories.shape[0]
        S = trajectories.shape[1]

        step_dt = np.array(self._datastore.step_length, dtype="timedelta64[h]")

        per_sample = []
        for b in range(B):
            # Prediction: one DataArray per member -> concat along
            # ensemble_member, then expand init_time into a dim.
            member_das = []
            for s in range(S):
                scaled = trajectories[b, s] * self.state_std + self.state_mean
                da_m = self._create_dataarray_from_tensor(
                    tensor=scaled,
                    time=time_rows[b],
                    split="test",
                    category="state",
                )
                if isinstance(self._datastore, BaseRegularGridDatastore):
                    da_m = self._datastore.unstack_grid_coords(da_m)
                t0 = da_m.coords["time"].values[0] - step_dt
                da_m.coords["init_time"] = t0
                da_m.coords["lead_time"] = da_m.time - t0
                da_m = da_m.swap_dims({"time": "lead_time"}).drop_vars(
                    "time", errors="ignore"
                )
                member_das.append(da_m)

            ens_dim = xr.DataArray(
                np.arange(S, dtype=np.int64),
                dims=("ensemble_member",),
                name="ensemble_member",
            )
            da_pred = xr.concat(member_das, dim=ens_dim)
            da_pred = da_pred.expand_dims("init_time")
            da_pred.name = "prediction"

            # Target: single DataArray, expand init_time into a dim.
            target_scaled = target_states[b] * self.state_std + self.state_mean
            da_t = self._create_dataarray_from_tensor(
                tensor=target_scaled,
                time=time_rows[b],
                split="test",
                category="state",
            )
            if isinstance(self._datastore, BaseRegularGridDatastore):
                da_t = self._datastore.unstack_grid_coords(da_t)
            t0_t = da_t.coords["time"].values[0] - step_dt
            da_t.coords["init_time"] = t0_t
            da_t.coords["lead_time"] = da_t.time - t0_t
            da_t = da_t.swap_dims({"time": "lead_time"}).drop_vars(
                "time", errors="ignore"
            )
            da_t = da_t.expand_dims("init_time")
            da_t.name = "target"

            per_sample.append(
                xr.Dataset({"prediction": da_pred, "target": da_t})
            )

        ds_batch = xr.concat(per_sample, dim="init_time")

        if first_write:
            n_feat = ds_batch.sizes["state_feature"]
            n_lon = ds_batch.sizes["longitude"]
            n_lat = ds_batch.sizes["latitude"]
            n_lead = ds_batch.sizes["lead_time"]
            encoding = {
                "init_time": {
                    "units": "Seconds since 1970-01-01 00:00:00",
                    "dtype": "int64",
                },
                "prediction": {
                    # (init_time, ensemble_member, lead_time,
                    #  state_feature, longitude, latitude)
                    "chunks": (1, 1, n_lead, n_feat, n_lon, n_lat),
                },
                "target": {
                    # (init_time, lead_time, state_feature,
                    #  longitude, latitude)
                    "chunks": (1, n_lead, n_feat, n_lon, n_lat),
                },
            }
            os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
            logger.info(
                f"Initializing test-set ensemble zarr at {zarr_path} "
                f"(first batch with {B} init_times)"
            )
            ds_batch.to_zarr(
                zarr_path, mode="w", consolidated=True, encoding=encoding
            )
        else:
            logger.info(
                f"Appending {B} init_times to test-set ensemble zarr "
                f"at {zarr_path}"
            )
            ds_batch.to_zarr(zarr_path, mode="a", append_dim="init_time")

    def _resolve_forecasts_zarr_path(self) -> str:
        """Path of the combined test-set ensemble zarr store."""
        path = getattr(self.args, "forecasts_zarr_path", None)
        if path:
            return path
        return os.path.join(
            self.logger.save_dir,
            "ensemble_forecasts.zarr",
        )

    def on_test_epoch_start(self) -> None:
        """Reset the test-set ensemble zarr flag before a new test run."""
        self._forecasts_zarr_initialized = False

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        super().test_step(batch, batch_idx)

        (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_step(batch)

        self.test_metrics["spread_squared"].append(spread_squared_batch)
        self.test_metrics["ens_mse"].append(ens_mse_batch)

        # Compute additional ensemble metrics
        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(trajectories, dim=1)
        # (B, pred_steps, num_grid_nodes, d_f)

        # Compute MAE for ensemble mean + ensemble CRPS
        ens_maes = metrics.mae(
            ens_mean,
            target_states,
            ens_std,
            mask=self.loss_mask,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["ens_mae"].append(ens_maes)
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            None,
            mask=self.loss_mask,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["crps_ens"].append(crps_batch)

        # Save full test-set ensemble forecasts (all batches) to a single
        # zarr store, appended along init_time. Rank 0 only.
        if self.trainer.is_global_zero and getattr(
            self.args, "save_forecasts", False
        ):
            time = batch[-1]
            zarr_path = self._resolve_forecasts_zarr_path()
            self._save_ensemble_batch_to_zarr(
                time_rows=time,
                trajectories=trajectories,
                target_states=target_states,
                zarr_path=zarr_path,
                first_write=not self._forecasts_zarr_initialized,
            )
            self._forecasts_zarr_initialized = True

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                trajectories.shape[0],
                self.n_example_pred - self.plotted_examples,
            )

            if getattr(self.args, "save_ensemble_zarr", False):
                time = batch[-1]
                for i in range(n_additional_examples):
                    self._save_ensemble_example_to_zarr(
                        time[i],
                        trajectories[i],
                        target_states[i],
                        self.plotted_examples + i + 1,
                    )

            self.plot_ensemble_examples(
                batch,
                n_additional_examples,
                prediction=trajectories,
                split="test",
            )
            self.plotted_examples += n_additional_examples

    def plot_examples(self, batch, n_examples, split, prediction=None):
        # We don't want to plot examples for a single ensemble member.
        pass

    def plot_ensemble_examples(
        self,
        batch,
        n_examples,
        split="train",
        prediction=None,
        time_steps=None,
        log=True,
    ):
        """
        Plot ensemble forecast + mean and std
        (split argument should be unused, only for compatibility with ARModel)
        """
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            time,
        ) = batch

        if prediction is None:
            trajectories = self.sample_trajectories(
                init_states,
                forcing,
                boundary_forcing,
                atmosphere_forcing,
            )
            # (B, S, pred_steps, num_grid_nodes, d_f)
        else:
            trajectories = prediction
            # (B, S, pred_steps, num_grid_nodes, d_f)

        # Rescale to original data scale
        traj_rescaled = trajectories * self.state_std + self.state_mean
        target_rescaled = target_states * self.state_std + self.state_mean

        # Compute mean and std of ensemble
        ens_mean = torch.mean(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        # Iterate over the examples
        for example_i, (
            traj_slice,
            target_slice,
            ens_mean_slice,
            ens_std_slice,
            time_slice,
        ) in enumerate(
            zip(
                traj_rescaled[:n_examples],
                target_rescaled[:n_examples],
                ens_mean[:n_examples],
                ens_std[:n_examples],
                time[:n_examples],
            ),
            start=1,
        ):
            # Create xarray for plotting
            da_samples = [
                self._create_dataarray_from_tensor(
                    tensor=traj_slice[i, ...],
                    time=time_slice,
                    split=split,
                    category="state",
                ).unstack("grid_index")
                for i in range(traj_slice.shape[0])
            ]

            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            da_ens_mean = self._create_dataarray_from_tensor(
                tensor=ens_mean_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            da_ens_std = self._create_dataarray_from_tensor(
                tensor=ens_std_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")
            # traj_slice is (S, pred_steps, num_grid_nodes, d_f)
            # others are (pred_steps, num_grid_nodes, d_f)

            # Note: min and max values can not be in ensemble mean
            var_vmin = (
                torch.minimum(
                    traj_slice.flatten(0, 2).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    traj_slice.flatten(0, 2).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            if time_steps is None:
                # If no set of time steps given, iterate over all
                # prediction horizon time steps
                time_steps = list(range(1, len(target_slice) + 1))
            else:
                time_steps = list(time_steps)

            # Only plot selected (variable_index -> lead steps).
            # When empty, keep previous behavior (all vars × all steps).
            plot_filter = getattr(self.args, "var_leads_val_plot", None) or {}
            if plot_filter:
                allowed_t = set(time_steps) & {
                    s for steps in plot_filter.values() for s in steps
                }
                time_steps = [t for t in time_steps if t in allowed_t]

            plot_dict = {}
            for t_i in time_steps:
                time_title_part = (
                    f"t={t_i} ({self._datastore.step_length * t_i} h)"
                )
                var_names_list = self._datastore.get_vars_names("state")
                var_units_list = self._datastore.get_vars_units("state")
                var_items = list(
                    enumerate(zip(var_names_list, var_units_list, var_vranges))
                )
                if plot_filter:
                    var_items = [
                        (var_i, (var_name, var_unit, var_vrange))
                        for var_i, (var_name, var_unit, var_vrange) in var_items
                        if var_i in plot_filter and t_i in plot_filter[var_i]
                    ]

                # Create one figure per selected variable at this time step
                var_figs = {
                    var_name: vis.plot_ensemble_prediction(
                        [
                            da_samples[i].isel(
                                state_feature=var_i, time=t_i - 1
                            )
                            for i in range(traj_slice.shape[0])
                        ],
                        da_target.isel(state_feature=var_i, time=t_i - 1),
                        da_ens_mean.isel(state_feature=var_i, time=t_i - 1),
                        da_ens_std.isel(state_feature=var_i, time=t_i - 1),
                        self._datastore,
                        title=f"{var_name} ({var_unit}), {time_title_part}",
                        vrange=var_vrange,
                    )
                    for var_i, (var_name, var_unit, var_vrange) in var_items
                }

                if log:
                    # Log immediately
                    example_title = f"example_{example_i}"
                    wandb.log(
                        {
                            f"{var_name}_{example_title}": wandb.Image(fig)
                            for var_name, fig in var_figs.items()
                        }
                    )
                else:
                    # Store to return
                    plot_dict.update(
                        {
                            f"{var_name}_step_{t_i}_ex{example_i}": wandb.Image(
                                fig
                            )
                            for var_name, fig in var_figs.items()
                        }
                    )

                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

        return plot_dict

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch. Will
        gather stored tensors and perform plotting and logging on rank 0.
        """
        super().on_test_epoch_end()
        self.log_spsk_ratio(self.test_metrics, "test")

    def log_spsk_ratio(self, metric_vals, prefix):
        """
        Compute the mean spread-skill ratio for logging in evaluation

        metric_vals: dict with all metric values
        prefix: string, prefix to use for logging
        """
        # Compute mean spsk_ratio
        spread_squared_tensor = self.all_gather_cat(
            torch.cat(metric_vals["spread_squared"], dim=0)
        )  # (N_eval, pred_steps, d_f)
        ens_mse_tensor = self.all_gather_cat(
            torch.cat(metric_vals["ens_mse"], dim=0)
        )  # (N_eval, pred_steps, d_f)

        # Do not log during sanity check?
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            # Note that spsk_ratio is scale-invariant, so do not have to rescale
            spread = torch.sqrt(torch.mean(spread_squared_tensor, dim=0))
            skill = torch.sqrt(torch.mean(ens_mse_tensor, dim=0))
            # Both (pred_steps, d_f)

            # Include finite sample correction
            spsk_ratios = np.sqrt(
                (self.ensemble_size + 1) / self.ensemble_size
            ) * (
                spread / skill
            )  # (pred_steps, d_f)

            # Strip density channels before plotting/logging
            if self.use_density:
                spsk_ratios = spsk_ratios[:, : self._density_idx]

            log_dict = self.create_metric_log_dict(
                spsk_ratios, prefix, "spsk_ratio"
            )

            log_dict[f"{prefix}_mean_spsk_ratio"] = torch.mean(
                spsk_ratios
            )  # log mean
            wandb.log(log_dict)
