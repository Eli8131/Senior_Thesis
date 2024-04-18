import numpy as np
import xarray as xr
import zarr
import dask
import sys
import time
import spectrum
from scipy import signal
import analytic_wavelet

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

# wavelet function for filtering
def wavefiltp(psif,z):
    zp = 1/np.sqrt(2)*analytic_wavelet.analytic_wavelet_transform(z, psif, False)
    return zp

# create filter parameters and boundaries
tidefreq = 24./np.array([12.4206012 ,12, 23.93447213, 24.06588766, 11.96723606, 12.65834751, 25.81933871, 26.868350]);
# M2, S2, K1, O1, K2, N2, P1, Q1
dt = 1/24
gamma = 3
bw = 0.3

# M2
P2 = tidefreq[0]/bw
beta2 = P2**2/gamma
omega2 = 2*np.pi*tidefreq[0]*dt 
# K1
P1 = tidefreq[2]/bw
beta1 = P1**2/gamma
omega1 = 2*np.pi*tidefreq[2]*dt 

#compute the wavelet filters
morse2 = analytic_wavelet.GeneralizedMorseWavelet(gamma, beta2)
psi2, psif2 = morse2.make_wavelet(1440,omega2)

morse1 = analytic_wavelet.GeneralizedMorseWavelet(gamma, beta1)
psi1, psif1 = morse1.make_wavelet(1440,omega1)

for i in list11:
    # load 'fast and deep' ID list
    ids = np.loadtxt('/home/exf512/data/lag_0m_hycom/deep_fast/ID_lists/deep_fast_IDs_'+str(
    int(i))).astype(int)
    
    # load lagrangian dataset
    ds = xr.open_dataset('/projectnb/msldrift/pra-drifters/hycom/global_hycom_step_'+str(
        int(i))+'_withmean_chunked.zarr', engine = 'zarr')

    # chunk dataset
    ds = ds.chunk({'obs':1440, 'trajectory':18351})

    # create complex velocity array
    cv0 = ds["ve"][ids,:]+1j*(ds["vn"][ids,:])

    cv0 = cv0.chunk(10000, 1440)

    # calculate positive and negative rotational energy 
    zp2 = wavefiltp(psif2.flatten(),cv0)
    zn2 = np.conj(wavefiltp(psif2.flatten(),np.conj(cv0)))
    zp1= wavefiltp(psif1.flatten(),cv0)
    zn1 = np.conj(wavefiltp(psif1.flatten(),np.conj(cv0)))
    
    # create and save datasets
    dsout1 = xr.Dataset(data_vars=dict(zp=(["trajectory","obs"],dask.array.from_array(zp1,chunks='auto')),
                                 zn=(["trajectory","obs"],dask.array.from_array(zn1,chunks='auto')),
                                 lon=(["trajectory","obs"],ds.lon[ids,:].data),
                                 lat=(["trajectory","obs"],ds.lat[ids,:].data),
                                 time=(["trajectory","obs"],ds.time[ids,:].data),),
                   coords=dict(trajectory=(["trajectory"],ds.trajectory[ids].data),
                               obs=(["obs"],ds.obs.data),
                   )
                  )
    dsout1 = dsout1.chunk('auto')
    dsout2 = xr.Dataset(data_vars=dict(zp=(["trajectory","obs"],dask.array.from_array(zp2,chunks=(18351,1440))),
                                 zn=(["trajectory","obs"],dask.array.from_array(zn2,chunks=(18351,1440))),
                                 lon=(["trajectory","obs"],ds.lon[ids,:].data),
                                 lat=(["trajectory","obs"],ds.lat[ids,:].data),
                                 time=(["trajectory","obs"],ds.time[ids,:].data),),
                   coords=dict(trajectory=(["trajectory"],ds.trajectory[ids].data),
                               obs=(["obs"],ds.obs.data),
                   )
                  )
    dsout2=dsout2.chunk('auto')

    dsout1.to_zarr('/projectnb/msldrift/tidaldrift/faigle/filtered_wavelet/D/hycom_D_0m_'+str(int(i))+'_zpzn.zarr')
    dsout2.to_zarr('/projectnb/msldrift/tidaldrift/faigle/filtered_wavelet/SD/hycom_SD_0m_'+str(int(i))+'_zpzn_03.zarr')




