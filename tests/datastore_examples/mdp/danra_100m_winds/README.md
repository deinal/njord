# Cropped DANRA MDP example

This directory contains an [mllam-data-prep](https://github.com/mllam/mllam-data-prep)
config for a small DANRA weather extract (used historically with Neural-LAM).

Njord CI does **not** run against this dataset: the ocean models need per-variable
bathymetry masks that this example does not provide, and the npy/MEPS datastore
has been removed. Tests use `tests/dummy_datastore.py` instead.
