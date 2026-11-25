# Baltic Sea data configs

Configs for the Njord-Baltic models in [`njord_baltic_models`](../njord_baltic_models/README.md).
All datasets are built with [mllam-data-prep](https://github.com/mllam/mllam-data-prep) on a Lambert conformal projection (central longitude 20°, central latitude 60°) at a daily time step.
The `path` entries point to machine-specific zarr stores and need to be changed to your own data.

## Neural-LAM configs

These are passed as `--config_path`. Graphs are read from `data/graphs/`.

| Config | Use | Ocean state | Boundary | Atmosphere | Statistics |
|--------|-----|-------------|----------|------------|------------|
| `baltic_sea.yaml` | Pretraining | `baltic_sea_mdp_config` | `baltic_boundary_mdp_config` | `baltic_era5_mdp_config` | own |
| `baltic_sea_density.yaml` | Pretraining, with ice density channel | `baltic_sea_mdp_config` | `baltic_boundary_mdp_config` | `baltic_era5_mdp_config` | own |
| `baltic_sea_finetune.yaml` | Finetuning on analysis | `baltic_sea_ft_mdp_config` | `baltic_boundary_ft_mdp_config` | `baltic_era5_ft_mdp_config` | reanalysis |
| `baltic_sea_forecasts.yaml` | Forecast evaluation | `baltic_sea_gt_forecasts_mdp_config` | `baltic_boundary_forecasts_mdp_config` | `baltic_ifs_forecasts_mdp_config` | reanalysis |

All configs weight `sla`, `siconc`, `sithick` by 0.5 and the depth-level variables by 0.2 in the loss, and clamp `siconc` to `[0, 1]` and `sithick` to `≥ 0`.
`density_channel` (all configs except `baltic_sea.yaml`) adds a binary ice-presence channel (1 where `siconc > 0`) to the state; during rollout `siconc` and `sithick` are set to zero where the predicted density is below 0.5.
The `statistics_*` entries make finetuning and evaluation normalise with the reanalysis mean/std, so they match pretraining.

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

### Variables

- **State** (23): `sla`, `siconc`, `sithick`, and `thetao`, `so`, `uo`, `vo` at 1, 9, 28, 47, 91 m. Day-of-year sin/cos are added as forcing.
- **Static**: `deptho`, `mdt`, `x_coord`, `y_coord`, `coast_dist`, plus per-variable sea masks.
- **Boundary**: `zos`, and `thetao`, `so`, `uo`, `vo` at 0, 47, 92 m.
- **Atmosphere**: `sotemair` (2 m air temperature), `sowinu10` / `sowinv10` (10 m wind), `sosudosw` / `sosudolw` (downward short-/longwave radiation), `sowaprec` (precipitation), `sod2m` (2 m dewpoint), `somslpre` (mean sea level pressure).

### Building the zarrs

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

## Graphs

`graphs/` contains the prebuilt 3-level meshes over Baltic sea points, with overview plots:

- `baltic_hierarchical`: hierarchical quadrilateral mesh, used by the SeaCast (`graph_fm`) checkpoints.
- `baltic_cluster`: cluster (k-means) mesh, used by the Njord-Baltic (`graph_efm`) checkpoints.

They can be rebuilt with `python -m neural_lam.create_graph --config_path data/baltic_sea.yaml ...` (see the main README).
