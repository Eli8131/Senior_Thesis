import numpy as np
import xarray as xr
import zarr
import dask
import sys
import time
st = time.time()
import clouddrift
from clouddrift.wavelet import morse_wavelet, morse_wavelet_transform, wavelet_transform

from dask.distributed import Client
from dask_jobqueue import LSFCluster

dask.config.set({"distributed.scheduler.worker-saturation":1.0})

cluster = LSFCluster(cores=1,project="tidaldrift",walltime="72:00",queue="bigmem",memory="250GB")
time.sleep(5)
cluster.scale(10)
time.sleep(10)
client = Client(cluster)
time.sleep(5)
list11 = np.linspace(1,11,11)
# define filter parameters for lowpass
dt = 1/24
gamma = 3
beta = 0
L = 2 # 2-day cutoff for the lowpass
omega = np.array([1 / L]) * 2 * np.pi * dt

# filter function
def myfun(x,gamma,beta,omega,time_axis=-1):
    #try:
    wtp = morse_wavelet_transform(0.5*x,gamma,beta,omega,time_axis=time_axis)
    return wtp

for i in list11:
    # open eulerian dataset
    ds = xr.open_zarr('/scratch/tidaldrift/hycom_ss/hycom12_step_'+str(int(i))+'_15m_E3.zarr',chunks='auto')
    mke = np.full(ds["Latitude"].shape, np.nan)
    for j in range(mke.shape[0]):
        # create complex velocity array
        x = ds['u'][:,j,:] + 1j*(-ds["v"][:,j,:])
        mke_tmp = xr.apply_ufunc(myfun,x,gamma,beta,omega,
                                  input_core_dims=[["time"],[],[],[]],
                                  output_core_dims = [['time']],
                                  dask = "parallelized",
                                  output_dtypes=np.float64,
                                  #kwargs={'time_axis': -1}
                                )
        # calculate the mean kinetic energy
        mke1 = np.abs(mke_tmp.mean(dim='time'))**2
        mke[j,:] = mke1

        #mke[j,:] = eke_tmp
        if j % 10 == 0:
            print(j)

    np.save('/projectnb/msldrift/tidaldrift/faigle/eul_wavelet_new/LF_XY/eul_LF_mke_step_'+str(int(i))+'_15m_cw.npy', mke)
    
    # repeat for cc
    ds = xr.open_zarr('/scratch/tidaldrift/hycom_ss/hycom12_step_'+str(int(i))+'_15m_E3.zarr',chunks='auto')
    mke = np.full(ds["Latitude"].shape, np.nan)
    for j in range(mke.shape[0]):
        x = ds['u'][:,j,:] + 1j*(ds["v"][:,j,:])
        mke_tmp = xr.apply_ufunc(myfun,x,gamma,beta,omega,
                                  input_core_dims=[["time"],[],[],[]],
                                  output_core_dims = [['time']],
                                  dask = "parallelized",
                                  output_dtypes=np.float64,
                                  #kwargs={'time_axis': -1}
                                )

        mke1 = np.abs(mke_tmp.mean(dim='time'))**2
        mke[j,:] = mke1

        #mke[j,:] = eke_tmp
        if j % 10 == 0:
            print(j)

    np.save('/projectnb/msldrift/tidaldrift/faigle/eul_wavelet_new/LF_XY/eul_LF_mke_step_'+str(int(i))+'_15m_cc.npy', mke)

