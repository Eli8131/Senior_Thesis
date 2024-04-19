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
for i in list11:
    # load IDs for 'deep' and fast trajectories
    ids= np.loadtxt('/home/exf512/summer/lag_0m_hycom/deep_fast/ID_lists/deep_fast_IDs_'+str(
    int(i))).astype(int)
    
    # open lagrangian dataset
    ds= xr.open_dataset('/projectnb/msldrift/pra-drifters/global_hycom_step_'+str(
        int(i))+'_withmean_chunked.zarr', engine = 'zarr')
    
    # create complex velocity time series
    c = (ds.ve[ids,:]) + (1j*(ds.vn[ids,:]))
    
    # declare bins
    lon = np.linspace(-180,180, 360*2)
    lat = np.linspace(-90, 90, 180*2)
    
    # calculate variance (KE)
    var = sc.stats.binned_statistic_2d(ds[int(i)].lon[ids[int(i)],:].to_numpy().flatten(), 
                                       ds[int(i)].lat[ids[int(i)],:].to_numpy().flatten(),
                                       c[int(i)].to_numpy().flatten(),
                                       statistic = np.nanvar,
                                       bins= [lon,lat])
    # calculated observation density
    count = sc.stats.binned_statistic_2d(ds[int(i)].lon[ids[int(i)],:].to_numpy().flatten(), 
                                       ds[int(i)].lat[ids[int(i)],:].to_numpy().flatten(),
                                       c[int(i)].to_numpy().flatten(),
                                       statistic = 'count',
                                       bins= [lon,lat])
   
    # create and save dataset
    dvcs= xr.Dataset(data_vars=dict(var = (['lon','lat'], var[int(i)].statistic.real),
                                 density = (['lon','lat'], count[int(i)].statistic),),
                  coords = dict(lon = var[int(i)].x_edge[0:-1], lat = var[int(i)].y_edge[0:-1]),)

    dvcs.to_netcdf('/scratch/tidaldrift/lag_var_count_ds_0m/lag_var_count_0m_withmean'+str(int(i))+'.nc')
