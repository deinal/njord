# Data configs

Configs for the Njord models: the regional Baltic Sea models in [`njord_baltic_models`](../njord_baltic_models/README.md) and the global models in [`njord_global_models`](../njord_global_models/README.md).
All datasets are built with [mllam-data-prep](https://github.com/mllam/mllam-data-prep) at a daily time step.
The `path` entries are relative to the repository root and point to `njord_baltic_data/` (Baltic) and `njord_global_data/` (global), so run all commands from the repository root.

Neural-LAM configs (`*.yaml` without `_mdp_config`) are passed as `--config_path` and name the mllam-data-prep configs (`*_mdp_config.yaml`) to read.
Each mllam-data-prep config is built into `<config name>.zarr` next to it, and graphs are read from `data/graphs/`.
`statistics_*` entries make a config normalise with the mean/std of another datastore, so that finetuning and evaluation use the same normalisation as pretraining.
`density_channel` adds a binary ice-presence channel (1 where `siconc > 0`) to the state; during rollout `siconc` and `sithick` are set to zero where the predicted density is below 0.5.

# Baltic Sea

The Baltic grid uses a Lambert conformal projection (central longitude 20°, central latitude 60°).

## Neural-LAM configs

| Config | Use | Ocean state | Boundary | Atmosphere | Statistics |
|--------|-----|-------------|----------|------------|------------|
| `baltic_sea.yaml` | Pretraining | `baltic_sea_mdp_config` | `baltic_boundary_mdp_config` | `baltic_era5_mdp_config` | own |
| `baltic_sea_density.yaml` | Pretraining, with ice density channel | `baltic_sea_mdp_config` | `baltic_boundary_mdp_config` | `baltic_era5_mdp_config` | own |
| `baltic_sea_finetune.yaml` | Finetuning on analysis | `baltic_sea_ft_mdp_config` | `baltic_boundary_ft_mdp_config` | `baltic_era5_ft_mdp_config` | reanalysis |
| `baltic_sea_forecasts.yaml` | Forecast evaluation | `baltic_sea_gt_forecasts_mdp_config` | `baltic_boundary_forecasts_mdp_config` | `baltic_ifs_forecasts_mdp_config` | reanalysis |

All configs weight `sla`, `siconc`, `sithick` by 0.5 and the depth-level variables by 0.2 in the loss, and clamp `siconc` to `[0, 1]` and `sithick` to `≥ 0`.
All configs except `baltic_sea.yaml` use the density channel.

## mllam-data-prep configs

| Config | Content | Time | Splits |
|--------|---------|------|--------|
| `baltic_sea_mdp_config.yaml` | Baltic reanalysis | 1993–2024 | train 1993–2021, val 2022, test 2024 |
| `baltic_boundary_mdp_config.yaml` | Global ocean boundary | 1993–2024 | same as above |
| `baltic_era5_mdp_config.yaml` | ERA5 | 1993–2024 | same as above |
| `baltic_sea_ft_mdp_config.yaml` | Baltic analysis | 2023–2024 | train 2023, val 2024-01–06, test 2024-07–12 |
| `baltic_boundary_ft_mdp_config.yaml` | Global ocean boundary | 2023–2024 | same as above |
| `baltic_era5_ft_mdp_config.yaml` | ERA5 | 2023–2024 | same as above |
| `baltic_sea_gt_forecasts_mdp_config.yaml` | Baltic analysis as ground truth (`init_time` × `lead_time`) | 2024 | test, weekly `init_time` 2024-01-02 to 2024-12-24 |
| `baltic_boundary_forecasts_mdp_config.yaml` | Global ocean forecasts (`init_time` × `lead_time`) | 2024 | same as above |
| `baltic_ifs_forecasts_mdp_config.yaml` | IFS forecasts (`init_time` × `lead_time`) | 2024 | same as above |

## Variables

- **State** (23): `sla`, `siconc`, `sithick`, and `thetao`, `so`, `uo`, `vo` at 1, 9, 28, 47, 91 m. Day-of-year sin/cos are added as forcing.
- **Static**: `deptho`, `mdt`, `x_coord`, `y_coord`, `coast_dist`, plus per-variable sea masks.
- **Boundary**: `zos`, and `thetao`, `so`, `uo`, `vo` at 0, 47, 92 m.
- **Atmosphere**: `sotemair` (2 m air temperature), `sowinu10` / `sowinv10` (10 m wind), `sosudosw` / `sosudolw` (downward short-/longwave radiation), `sowaprec` (precipitation), `sod2m` (2 m dewpoint), `somslpre` (mean sea level pressure).

## Building the zarrs

Each command writes `<config name>.zarr` next to the config.

```bash
# Pretraining (1993–2024, daily)
# Baltic reanalysis: ocean state, static fields and sea masks
mllam_data_prep --config data/baltic_sea_mdp_config.yaml
# Global ocean reanalysis on the lateral boundary
mllam_data_prep --config data/baltic_boundary_mdp_config.yaml
# ERA5 atmospheric forcing
mllam_data_prep --config data/baltic_era5_mdp_config.yaml

# Finetuning (2023–2024, daily)
# Baltic analysis: ocean state, static fields and sea masks
mllam_data_prep --config data/baltic_sea_ft_mdp_config.yaml
# Global ocean boundary
mllam_data_prep --config data/baltic_boundary_ft_mdp_config.yaml
# ERA5 atmospheric forcing
mllam_data_prep --config data/baltic_era5_ft_mdp_config.yaml

# Forecast evaluation (weekly init_time in 2024, init_time x lead_time)
# Baltic analysis as ground truth
mllam_data_prep --config data/baltic_sea_gt_forecasts_mdp_config.yaml
# Global ocean forecasts on the lateral boundary
mllam_data_prep --config data/baltic_boundary_forecasts_mdp_config.yaml
# IFS atmospheric forecasts
mllam_data_prep --config data/baltic_ifs_forecasts_mdp_config.yaml
```

The pretraining zarrs are also needed for finetuning and evaluation, since their statistics are used for normalisation (`statistics_*` in the Neural-LAM configs).

# Global ocean

The global grids are regular latitude–longitude grids at 1° and 0.25°.
The 0.25° analysis has a different bathymetry (and sea mask) than the reanalysis, which is why it has its own graphs.

## Neural-LAM configs

| Config | Use | Ocean state | Atmosphere | Statistics |
|--------|-----|-------------|------------|------------|
| `global_ocean_1_density.yaml` | Pretraining at 1° and building 1° graphs, with ice density channel | `global_ocean_1_mdp_config` | `global_era5_1_mdp_config` | own |
| `global_ocean_1_4_density.yaml` | Finetuning at 0.25° and building 0.25° graphs, with ice density channel | `global_ocean_1_4_mdp_config` | `global_era5_1_4_mdp_config` | own |
| `analysis_ocean.yaml` | Finetuning on analysis | `analysis_ocean_mdp_config` | `analysis_era5_mdp_config` | 0.25° reanalysis |
| `analysis_ocean_forecasts.yaml` | Forecast evaluation | `analysis_ocean_forecasts_mdp_config` | `analysis_ifs_forecasts_mdp_config` | 0.25° reanalysis |

All configs use a latitude-weighted loss, weight `zos`, `siconc`, `sithick` by 0.5 and the depth-level variables by 0.17, and clamp `siconc` to `[0, 1]` and `sithick` to `≥ 0`.
All configs use the density channel. Graphs are built with the density configs, since graph building only uses the grid.

## mllam-data-prep configs

| Config | Content | Time | Splits |
|--------|---------|------|--------|
| `global_ocean_1_mdp_config.yaml` | 1° global reanalysis | 1993–2023 | train 1993–2021, val 2022, test 2023 |
| `global_era5_1_mdp_config.yaml` | ERA5 on the 1° grid | 1993–2023 | same as above |
| `global_ocean_1_4_mdp_config.yaml` | 0.25° global reanalysis | 1993–2023 | same as above |
| `global_era5_1_4_mdp_config.yaml` | ERA5 on the 0.25° grid | 1993–2023 | same as above |
| `analysis_ocean_mdp_config.yaml` | 0.25° global analysis | 2023–2024 | train 2023, val 2024-01–06, test 2024-07–12 |
| `analysis_era5_mdp_config.yaml` | ERA5 on the analysis grid | 2023–2024 | same as above |
| `analysis_ocean_forecasts_mdp_config.yaml` | 0.25° global analysis as ground truth (`init_time` × `lead_time`) | 2024 | test, weekly `init_time` 2024-01-02 to 2024-12-24 |
| `analysis_ifs_forecasts_mdp_config.yaml` | IFS forecasts on the analysis grid (`init_time` × `lead_time`) | 2024 | same as above |

## Variables

- **State** (27): `zos`, `siconc`, `sithick`, and `thetao`, `so`, `uo`, `vo` at 0, 47, 92, 222, 318, 541 m. Day-of-year sin/cos are added as forcing.
- **Static**: `deptho`, `mdt`, `sin_lon`, `cos_lon`, `sin_lat`, `cos_lat`, `coast_dist`, plus per-variable sea masks.
- **Atmosphere**: the same 8 variables as for the Baltic Sea.

## Building the zarrs

```bash
# Pretraining at 1° (1993–2023, daily)
mllam_data_prep --config data/global_ocean_1_mdp_config.yaml
mllam_data_prep --config data/global_era5_1_mdp_config.yaml

# Finetuning at 0.25° (1993–2023, daily)
mllam_data_prep --config data/global_ocean_1_4_mdp_config.yaml
mllam_data_prep --config data/global_era5_1_4_mdp_config.yaml

# Finetuning on analysis (2023–2024, daily)
mllam_data_prep --config data/analysis_ocean_mdp_config.yaml
mllam_data_prep --config data/analysis_era5_mdp_config.yaml

# Forecast evaluation (weekly init_time in 2024, init_time x lead_time)
mllam_data_prep --config data/analysis_ocean_forecasts_mdp_config.yaml
mllam_data_prep --config data/analysis_ifs_forecasts_mdp_config.yaml
```

The 0.25° reanalysis zarrs are also needed for analysis finetuning and evaluation, since their statistics are used for normalisation.

# Graphs

The prebuilt graphs are not stored in git. Download them into `graphs/` from [deinal/njord-graphs](https://huggingface.co/datasets/deinal/njord-graphs).
[`graphs/README.md`](graphs/README.md) lists which model uses which graph and has the commands to rebuild them.
