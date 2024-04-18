import numpy as np
import xarray as xr
import zarr
import dask
import sys
import time

import scipy as sc
from dask.distributed import Client
from dask_jobqueue import LSFCluster

dask.config.set({"distributed.scheduler.worker-saturation":1.0})

cluster = LSFCluster(cores=1,project="tidaldrift",walltime="72:00",queue="normal",memory="250GB")
time.sleep(5)
cluster.scale(10)
time.sleep(10)
client = Client(cluster)
time.sleep(5)

list11 = np.linspace(1, 11, 11)
lon = np.linspace(-180,180, 360*2)
lat = np.linspace(-90, 90, 180*2)

for i in list11:
    # open eulerian dataset and create complex velocity array
    ds = xr.open_zarr('/scratch/tidaldrift/hycom_ss/hycom12_step_'+str(int(i))+'_15m_E3.zarr/')
    x = ds['u'] + 1j*(ds["v"])
    # calculate variance
    a = np.nanvar(x, axis = 0)
    # average in 1/2 degree lon and lat bins
    var = sc.stats.binned_statistic_2d(ds.Longitude.to_numpy().flatten(), 
                                   ds.Latitude.to_numpy().flatten(),
                                   a.flatten(),
                                   statistic = np.nanmean,
                                   bins= [lon,lat])
    # create and save dataset
    dvcs = xr.Dataset(data_vars=dict(var = (['lon','lat'], var.statistic.real),),
              coords = dict(lon = var.x_edge[0:-1], lat = var.y_edge[0:-1]),)
    dvcs.to_netcdf('/projectnb/msldrift/tidaldrift/faigle/eul_total_var/eul_var_15m_step_'+str(int(i))+'.nc')
    
    # repeat process at 0 m
    ds = xr.open_zarr('/scratch/tidaldrift/hycom_ss/hycom12_step_'+str(int(i))+'_0m_E3.zarr/')
    x = ds['u'] + 1j*(ds["v"])
    a = np.nanvar(x, axis = 0)

    var = sc.stats.binned_statistic_2d(ds.Longitude.to_numpy().flatten(), 
                                   ds.Latitude.to_numpy().flatten(),
                                   a.flatten(),
                                   statistic = np.nanmean,
                                   bins= [lon,lat])
    dvcs = xr.Dataset(data_vars=dict(var = (['lon','lat'], var.statistic.real),),
              coords = dict(lon = var.x_edge[0:-1], lat = var.y_edge[0:-1]),)
    dvcs.to_netcdf('/projectnb/msldrift/tidaldrift/faigle/eul_total_var/eul_var_0m_step_'+str(int(i))+'.nc')


