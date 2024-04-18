import numpy as np
import xarray as xr
import zarr
import dask
import sys
import time
import spectrum
import analytic_wavelet
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

#Global Variables
list11 = np.linspace(1, 11, 11)
dt = 1/24
L=2
lon = np.linspace(-180,180, 360*2)
lat = np.linspace(-90, 90, 180*2)

for i in list11:
    # load lagrangian traj IDs for deep and fast drifters
    ids = np.loadtxt('/home/exf512/summer/lag_15m_hycom/deep_fast_IDs_15m/deep_fast_IDs_15m'+str(
    int(i))).astype(int)
    
    # load lagrangian dataset
    ds = xr.open_dataset('/projectnb/msldrift/pra-drifters/global_hycom_15m_step_'+str(int(i))+'_withmean_chunked.zarr', engine = 'zarr')
    
    # chunk dataset
    ds = ds.chunk({'obs':1440, 'trajectory':18351})
    
    # create complex velocity array
    cv0 = (ds.ve[ids,:]) + (1j*(ds.vn[ids,:]))

    # create lowpass filter parameters
    w = (1/L)/((1/dt)/2)
    b,a = sc.signal.butter(3,w,'low')
    cvf0 = sc.signal.filtfilt(b,a,cv0)
    
    # calculate variance 
    var = sc.stats.binned_statistic_2d(ds.lon[ids,:].to_numpy().flatten(), 
                                   ds.lat[ids,:].to_numpy().flatten(),
                                   cvf0.flatten(),
                                   statistic = np.nanvar,
                                   bins= [lon,lat])
    
    # create and save dataset
    dvcs = xr.Dataset(data_vars=dict(var = (['lon','lat'], var.statistic.real),),
              coords = dict(lon = var.x_edge[0:-1], lat = var.y_edge[0:-1]),)

    dvcs.to_netcdf('/projectnb/msldrift/tidaldrift/faigle/LF_var_count_ds_15m/LF_var_count_15m_'+str(int(i))+'_withmean.nc')

    # repeat process for 15 meters depth
for i in list11:
    # load lagrangian traj IDs for deep and fast drifters
    ids = np.loadtxt('/home/exf512/summer/lag_0m_hycom/deep_fast/ID_lists/deep_fast_IDs_'+str(
    int(i))).astype(int)
    
    # load lagrangian dataset
    ds = xr.open_dataset('/projectnb/msldrift/pra-drifters/global_hycom_step_'+str(
        int(i))+'_withmean_chunked.zarr', engine = 'zarr')
    
    # chunk dataset
    ds = ds.chunk({'obs':1440, 'trajectory':18351})
    
    # create complex velocity array
    cv0 = (ds.ve[ids,:]) + (1j*(ds.vn[ids,:]))
    
    # create lowpass filter parameters
    w = (1/L)/((1/dt)/2)
    b,a = sc.signal.butter(3,w,'low')
    cvf0 = sc.signal.filtfilt(b,a,cv0)
    
    # calculate variance
    var = sc.stats.binned_statistic_2d(ds.lon[ids,:].to_numpy().flatten(), 
                                   ds.lat[ids,:].to_numpy().flatten(),
                                   cvf0.flatten(),
                                   statistic = np.nanvar,
                                   bins= [lon,lat])

    # create and save dataset
    dvcs = xr.Dataset(data_vars=dict(var = (['lon','lat'], var.statistic.real),),
              coords = dict(lon = var.x_edge[0:-1], lat = var.y_edge[0:-1]),)

    dvcs.to_netcdf('/projectnb/msldrift/tidaldrift/faigle/LF_var_count_ds_0m/LF_var_count_0m_'+str(int(i))+'_withmean.nc')
