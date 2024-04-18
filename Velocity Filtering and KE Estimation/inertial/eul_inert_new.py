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
from clouddrift.sphere import coriolis_frequency
from concurrent.futures import ProcessPoolExecutor

from dask.distributed import Client
from dask_jobqueue import LSFCluster

dask.config.set({"distributed.scheduler.worker-saturation":1.0})

cluster = LSFCluster(cores=1,project="tidaldrift",walltime="96:00",queue="bigmem",memory="250GB")
time.sleep(5)
cluster.scale(10)
time.sleep(10)
client = Client(cluster)
time.sleep(5)

list11 = np.linspace(2,11,10)
# define a function that extracts inertial velocity based on latitude
# trying to mask inside this function to be used by xr.apply_ufunc seems fraught
def inertial_velocity(u,v,lat,time_step=1/24,relative_bandwidth=0.15,time_axis=0):
    try:
        gamma = 3
        cf = np.abs(coriolis_frequency(lat)*60*60*24/(2*np.pi))
        P = cf/relative_bandwidth
        beta = P**2/gamma
        omega = 2*np.pi*np.array([cf])*time_step
        # replace NaN with zeros
        ui = np.where(np.isnan(u), 0, u)
        vi = np.where(np.isnan(v), 0, v)
        # if SH, inertial oscillations are positive otherwise negative in NH
        if lat < 0:
            wtp = morse_wavelet_transform(0.5*(ui+1j*vi),
                                          gamma,beta,omega,time_axis=time_axis)
        else:        
            wtp = morse_wavelet_transform(0.5*(ui-1j*vi),
                                          gamma,beta,omega,time_axis=time_axis)
    except:
        wtp = np.full(u.shape[::-1],np.nan)*(1+1j*1)
    return wtp
# now do without dask, using processpoolexecutor maybe
def get_inertial_eke(u,v,lat):
    relative_bandwidth = 0.15
    time_step = 1/24
    gamma = 3
    cf = np.abs(coriolis_frequency(lat)*60*60*24/(2*np.pi))
    P = cf/relative_bandwidth
    beta = P**2/gamma
    omega = 2*np.pi*np.array([cf])*time_step
    try:
        # if SH, inertial oscillations are positive otherwise negative in NH
        if lat < 0:
            wtp = morse_wavelet_transform(0.5*(u+1j*v),gamma,beta,omega,time_axis=0)
        else:        
            wtp = morse_wavelet_transform(0.5*(u-1j*v),gamma,beta,omega,time_axis=0)
        eke = (np.abs(wtp)**2).mean()
    except:
        print('except')
        eke = np.nan
    return eke
for i in list11:
    #open subsampled eulerian dataset
    ds = xr.open_zarr('/scratch/tidaldrift/hycom_ss/hycom12_step_'+str(int(i))+'_15m_E3.zarr',chunks="auto")
    #create empty array for loop
    eke = np.full(ds["Latitude"].shape, np.nan)
    # loop over the Y dimension
    for j in range(eke.shape[0]):
        if len(np.unique(ds["Latitude"][j,:])) == 1:
            # find the land points; these are replaced by zeros inside the function
            mask = np.isnan(ds["u"][:,j,:]).all(axis=0)
            wtp = xr.apply_ufunc(inertial_velocity,ds["u"][:,j,:],ds["v"][:,j,:],ds["Latitude"][j,0].values,
                                      input_core_dims=[["X"],["X"],[]],
                                      output_core_dims = [["X"]],
                                      dask = "parallelized",
                                      output_dtypes=np.complex128  
                                    ).compute()
            eke[j,:] = (np.abs(wtp)**2).mean(dim="time")
            # re-establish the land points
            eke[j,mask] = np.nan
        else:
            with ProcessPoolExecutor() as executor:
                eke_row = list(executor.map(get_inertial_eke, ds["u"][:,j,:].to_numpy().transpose(),
                                            ds["v"][:,j,:].to_numpy().transpose(),ds["Latitude"][j,:].to_numpy().transpose()))
                eke[j,:] = eke_row 
        if j % 10 == 0:
            print(j)
    #save file        
    np.save('/scratch/tidaldrift/eul_inert_new/eul_inert_eke_step_'+str(int(i))+'_15m_v2.npy', eke)
