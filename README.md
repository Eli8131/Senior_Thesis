# symposium_2024
This repository contains examples of the code used to filter datasets by tidal band and estimate oceanic kinetic energy. The resulting figures are also here.

# File List:
`Spectral Analysis`: This directory holds all Python scripts and figures related to globally-averaged spectra and latitudinally-averaged spectra.

`Velocity Filtering and KE Estimation`: This directory holds all Python scripts used to filter the data around specific tidal bands (D, SD, LF, inertial) and estimate the oceanic KE in each band. KE maps and ratio figures comparing Eulerian and Lagrangian calculations are included.

### All below folders/files are contained within `Spectral Analysis` folder
`Global Spectra`: 

1) `calculate-hycom-lagrangian-spectra.py`: Used to calculate Lagrangian spectra.

2) `hycom_spectra_estimate.py`: Used to calculate Eulerian spectra

3) `eul_lag_spec.ipynb`: Used to create Eulerian and Lagrangian comparison figures.

4) `Figures`: Contains all Eulerian vs. Lagrangian global spectra plots at 0 and 15 meters.

`Latitudinal Spectra`:

1) `zonal_avg.py`: Used to calculate and plot Lagrangian and Eulerian latitudinally-averaged spectra.

2) `Figures`: Contains all figures created by `zonal_avg.py`.

### All below folders/files are contained within `Velocity Filtering and KE Estimation` folder


