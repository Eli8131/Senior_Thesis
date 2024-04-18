# code to calculate periodogram
import numpy as np
import xarray as xr
import zarr
import time
import xrft
from dask.distributed import Client
from dask_jobqueue import LSFCluster

cluster = LSFCluster(cores=1,project="tidaldrift",walltime="72:00",queue="bigmem",memory="300GB")
time.sleep(5)
print(cluster)

cluster.scale(10)
time.sleep(10)
print(cluster)

client = Client(cluster)
time.sleep(5)
print(client)

datadir = '/projects2/rsmas/tidaldrift/hycom_zarr/'

# with i between 1 and 10, the total length with be 1440
for i in range(11,12):
    ds1 = xr.open_zarr(datadir + 'hycom12-'+str(i)+'-rechunked-corr.zarr',consolidated=True)
    ds2 = xr.open_zarr(datadir + 'hycom12-'+str(i+1)+'-rechunked-corr.zarr',consolidated=True)
    ds12 = xr.combine_by_coords([ds1,ds2],data_vars=['u','v'],combine_attrs='drop_conflicts')

    # load the mean velocity fields
    mds = xr.open_dataset('/projectnb/msldrift/tidaldrift/hycom_analysis/hycom12-mean-uv.nc')

    # define and apply mask? When dshould it be applied?
    bathy = xr.open_dataset('/projectnb/msldrift/hycom/102_archv.bathy.nc')
    mask = bathy.bathymetry[0,:,:]>0
    
    # form the velocity anomalies DataArray
    uva = (ds12.u[0:1440,:,:,:]+1j*ds12.v[0:1440,:,:,:]-(mds.mean_u[:,:,:]+1j*mds.mean_v[:,:,:])).chunk({'time':1440})
    
    # some factors needed to accurately calculate spectra properly
    foo = np.ones(ds12['u'].shape[0])
    boxcar_fac = np.square(foo).sum()
    dt = 1/24
    
    # define fft object with xrft
    zfft = xrft.fft(uva,dim=['time'],window='boxcar',detrend=None,true_amplitude=False,shift=False,chunks_to_segments=False)
    
    # define spectra
    szz = (np.abs(zfft)**2)*dt/boxcar_fac
    
    # apply mask
    szz = szz.where(mask)
    
    # drop some variables
    szz = szz.drop_vars(['MT','Date'])
    
    # make an xarray dataset
    szz = szz.to_dataset(name='spectrum')
    
    # write result and compute?
    szz.to_zarr('/projects2/rsmas/tidaldrift/hycom_results/hycom12-spectrum-'+str(i)+'.zarr')
    
    ds1.close()
    ds2.close()
    ds12.close()
