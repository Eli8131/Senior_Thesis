import numpy as np
import xarray as xr
import zarr
import dask
import sys
import time
from scipy.stats import binned_statistic_2d
st = time.time()
import clouddrift
from clouddrift.wavelet import morse_wavelet, morse_wavelet_transform, wavelet_transform

from dask.distributed import Client
from dask_jobqueue import LSFCluster

dask.config.set({"distributed.scheduler.worker-saturation":1.0})

cluster = LSFCluster(cores=1,project="msldrift",walltime="96:00",queue="bigmem",memory="250GB")
time.sleep(5)
cluster.scale(10)
time.sleep(10)
client = Client(cluster)
time.sleep(5)

list11 = np.linspace(1,11,11)
# function to filter, as an example semi diurnal positive
# define filter parameters
tidefreq = 24./np.array([12.4206012 ,12, 23.93447213, 24.06588766, 11.96723606, 12.65834751, 25.81933871, 26.868350])
# M2, S2, K1, O1, K2, N2, P1, Q1
dt = 1/24
gamma = 3
bw = 0.3
# M2
P = tidefreq[0]/bw
beta = P**2/gamma
omega = 2*np.pi*np.array([tidefreq[0]])*dt # frequency must be defined as an np array

def myfun(u,v,gamma,beta,omega,time_axis=-1):
    #try:
    wtp = morse_wavelet_transform(0.5*(u+1j*v),gamma,beta,omega,time_axis=time_axis)
    var = np.nanmean(wtp.real**2,axis=time_axis) + np.nanmean(wtp.imag**2,axis=time_axis)
    #except:
    #var = np.nan
    return var

i = 1
# open subsampled eulerian dataset
ds = xr.open_zarr('/scratch/tidaldrift/hycom_ss/hycom12_step_'+str(int(i))+'_0m_E3.zarr',chunks="auto")
# create empty array for loop
eke = np.full(ds["Latitude"].shape, np.nan)
# apply_ufunc for myfun
# loop over the Y dimension
for j in range(eke.shape[0]):
    eke_tmp = xr.apply_ufunc(myfun,ds["u"][:,j,:],ds["v"][:,j,:],gamma,beta,omega,
                              input_core_dims=[["time"],["time"],[],[],[]],
                              output_core_dims = [[]],
                              dask = "parallelized",
                              output_dtypes=np.float64,
                              #kwargs={'time_axis': -1}
                            )
    eke[j,:] = eke_tmp

# save file        
np.save('/projectnb/msldrift/tidaldrift/faigle/eul_wavelet_new/SD_XY/eul_SD_eke_step_'+str(int(i))+'_0m_cc_03.npy', eke)
