# code to apply the inertial extraction to Lagrangian hycom trajectories
import time
import numpy as np
import xarray as xr
import zarr
import dask
from clouddrift.kinematics import inertial_oscillation_from_position, velocity_from_position

from dask.distributed import Client
from dask_jobqueue import LSFCluster

dask.config.set({"distributed.scheduler.worker-saturation":1.0})

cluster = LSFCluster(cores=1,project="tidaldrift",walltime="96:00",queue="bigmem",memory="250GB")
time.sleep(5)
cluster.scale(10)
time.sleep(10)
client = Client(cluster)
time.sleep(5)

st = time.time()

list11 = np.linspace(1, 11, 11)
# define a function that apply consecutively inertial_oscillation_from_position and velocity_from_position
def inertial_velocity(lon,lat,time,time_step=3600,relative_bandwidth=0.15):
    try:
        xhat, yhat = inertial_oscillation_from_position(lon,lat,time_step=time_step,relative_bandwidth=relative_bandwidth)
        ui, vi = velocity_from_position(xhat,yhat,time,coord_system="cartesian")
    except:
        ui = np.full(time.shape,np.nan)
        vi = np.full(time.shape,np.nan)
    return ui + 1j*vi


for i in list11:
    ids = np.loadtxt('/home/exf512/data/lag_0m_hycom/deep_fast/ID_lists/deep_fast_IDs_'+str(int(i))).astype(int)

    ds = xr.open_zarr('/projectnb/msldrift/pra-drifters/hycom/global_hycom_step_'+str(int(i))+'_withmean_chunked.zarr',chunks = {'obs':1440})

    # apply function to all trajectories
    uvi = xr.apply_ufunc(inertial_velocity,(ds["lon"][ids,:]),ds["lat"][ids,:],ds["time"][ids,:],
                              input_core_dims=[["obs"],["obs"],["obs"]],
                              output_core_dims = [["obs"]],
                              dask = "parallelized",
                              vectorize=True,
                              output_dtypes=np.complex128  
                            )

    # make an xarray dataset
    dsout = uvi.to_dataset(name="uvi")
    dsout = dsout.chunk(chunks = {'obs':1440, 'trajectory':10000})

    dsout.to_zarr('/scratch/tidaldrift/io_midpoint/global_hycom_step_'+str(int(i))+'_inertial_0m.zarr')
    print(i)

for i in list11:
    ids = np.loadtxt('/home/exf512/data/lag_15m_hycom/deep_fast_IDs_15m/deep_fast_IDs_15m'+str(int(i))).astype(int)

    ds = xr.open_zarr('/projectnb/msldrift/pra-drifters/hycom/global_hycom_15m_step_'+str(int(i))+'_withmean_chunked.zarr',chunks = {'obs':1440})

    # apply function to all trajectories
    uvi = xr.apply_ufunc(inertial_velocity,(ds["lon"][ids,:]),ds["lat"][ids,:],ds["time"][ids,:],
                              input_core_dims=[["obs"],["obs"],["obs"]],
                              output_core_dims = [["obs"]],
                              dask = "parallelized",
                              vectorize=True,
                              output_dtypes=np.complex128  
                            )

    # make an xarray dataset
    dsout = uvi.to_dataset(name="uvi")
    dsout = dsout.chunk(chunks = {'obs':1440, 'trajectory':10000})

    dsout.to_zarr('/scratch/tidaldrift/io_midpoint/global_hycom_step_'+str(int(i))+'_inertial_15m.zarr')
    print(i)

et = time.time()    
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')