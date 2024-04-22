# Symposium_2024
Currently a work in progress. This repository contains examples of the code used in Eli Faigle's 2024 Senior Thesis to filter datasets by tidal band and estimate oceanic kinetic energy. The resulting figures are also here. 

# File List:
`Spectral Analysis`: This directory holds all Python scripts and figures related to globally-averaged spectra and latitudinally-averaged spectra.

`Velocity Filtering and KE Estimation`: This directory holds all Python scripts used to filter the data around specific tidal bands (D, SD, LF, inertial) and estimate the oceanic KE in each band. KE maps and ratio figures comparing Eulerian and Lagrangian calculations are included.

Note: Figure files are centered around the 'step' for which they are named: 31-Jan-2014, 02-Mar-2014, 01-Apr-2014, 01-May-2014, 31-May-2014, 30-Jun-2014, 30-Jul-2014, 29-Aug-2014, 28-Sep-2014, 28-Oct-2014, 27-Nov-2014. All figures were produced at 0 m and 15 m depths.

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
`inertial`:

1) `eul_inert.py`: Used to filter around inertial frequency and estimate Eulerian KE.

2) `lag_inertial_filtering.py`: Used to filter around inertial frequency and estimate Lagrangian KE.

3) `Figures`: Eulerian, Lagrangian, and ratio figures for inertial KE.

`LF`

1) `lag_LF.py`: Used to apply a lowpass filter to the Lagrangian velocities.

2) `lag_LF_15m_stats2d.py`: Used to estimate KE and bin in 1/2 degree lat and lon bins from the output of `butter_withmean.py`.

3) `eul_LF_eke_cw.py`: Used to filter Eulerian data around LF eddy kinetic energy and estimate the 'clockwise' energy. This is combined with the 'counterclockwise' energy to get the total.

4) `eul_LF_mke_15m.py`: Used to filter Eulerian data around LF mean kinetic energy. This is combined with the EKE to get the total kinetic energy.

5) `Figures`: Contains all LF EKE, MKE, and TKE figures.

`SD_D`:

1) `eul_wavelet_cc.py`: Used to apply a bandpass filter to the Eulerian velocities around the D and SD bands and estimate the 'clockwise' KE. This is combined with the 'counterclockwise' energy to get the total.

2) `lag_wavelet_0m_D_SD.py`: Used to apply a bandpass filter to the Lagrangian velocities around the D and SD bands and estimate the KE.

3) `Figures`: Contains all SD and D figures.

`unfiltered`:

1) `eul_var.py`: Used to estimate unfiltered Eulerian KE.

2) `lag_var.py`: Used to estimate the unfiltered Lagrangian KE.

3) `Figures`: Contains all unfiltered figures.
