# Python code to calculate Lagrangian spectra in HYCOM
# Intermediate step is to rewrite the Lagrangian 

import numpy as np
import xarray as xr
import xrft
import zarr
import dask
import sys
import dask.distributed
import time

from scipy import stats
from dask.distributed import Client
from dask_jobqueue import LSFCluster

dask.config.set({"distributed.scheduler.worker-saturation":1.0})

cluster = LSFCluster(cores=1,project="tidaldrift",walltime="72:00",queue="normal",memory="200GB")
time.sleep(5)
cluster.scale(10)
time.sleep(10)
client = Client(cluster)
time.sleep(5)

N = 1440 # enforce chunk size for observation

# open the zarr store with the lagrangian trajectories at 0m , including velocities
for i in range(1,12):
    file1 = '/projectnb/msldrift/pra-drifters/global_hycom_step_'+str(i)+'_2d.zarr'
    ds1 = xr.open_dataset(file1,engine='zarr',chunks={"obs":N}) # set chunks along time dimension
    print('working on step '+str(i))
    sys.stdout.flush()
    
    # recast longitude in  range[-180,180]
    ds1["lon"] = ds1.lon.where(ds1.lon>-180,ds1.lon+360)
    ds1["lon"] = ds1.lon.where(ds1.lon<180,ds1.lon-360)

    # interpolate the mean u and v velocity, here regridded
    mds = xr.open_dataset('/projectnb/msldrift/tidaldrift/hycom_analysis/hycom12-mean-uv-regridded.nc',engine='netcdf4')

    mve = mds.mean_u.sel(Depth=0).interp(lat=ds1.lat,lon=ds1.lon,method='linear')
    mvn = mds.mean_v.sel(Depth=0).interp(lat=ds1.lat,lon=ds1.lon,method='linear')

    # add the mean velocity to the dataset and rewrite the dataset
    print('interpolating step '+str(i))
    sys.stdout.flush()
    ds1["mve"] = (['trajectory','obs'],(mve.drop(['lat','lon','Depth'])).chunk(ds1.chunks).data.astype('float32'))
    ds1["mvn"] = (['trajectory','obs'],(mvn.drop(['lat','lon','Depth'])).chunk(ds1.chunks).data.astype('float32'))

    fileout1 = '/projectnb/msldrift/pra-drifters/global_hycom_step_'+str(i)+'_withmean_chunked.zarr'
    ds1.to_zarr(fileout1,mode="w",compute=True)
    ds1.close()

    # reopen dataset to ensure chunks? Need to specify obs chunk?
    ds1 = xr.open_zarr(fileout1,chunks={"obs":N})

    # some factors needed to accurately calculate spectra
    foo = np.ones(N)
    boxcar_fac = np.square(foo).sum()
    dt = 1/24

    # define fft objects with xrft
    zfft = xrft.fft((ds1.ve+1j*ds1.vn)-(ds1.mve+1j*ds1.mvn),dim=['obs'],window='boxcar',detrend=None,true_amplitude=False,shift=False,chunks_to_segments=False)
    # define spectra
    szz = (np.abs(zfft)**2)*dt/boxcar_fac

    # turn xarray DataArray into Xarray Dataset and write to zarr
    print('calculating spectra for step '+str(i))
    sys.stdout.flush()
    szz = szz.to_dataset(name='spectrum')
    szz.to_zarr('/projects2/rsmas/tidaldrift/hycom_results/hycom12-uv-lagrangian-spectrum-'+str(i)+'.zarr',mode='w')
    print('bunch '+str(i)+' done!')
    sys.stdout.flush()