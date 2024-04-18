# import computational and plotting packages
import xarray as xr
import numpy as np
import zarr
import time
import clouddrift
import dask
import matplotlib.pyplot as plt
# open dask cluster
from dask.distributed import Client
from dask_jobqueue import LSFCluster

dask.config.set({"distributed.scheduler.worker-saturation":1.0})

cluster = LSFCluster(cores=1,project="tidaldrift",walltime="72:00",queue="bigmem",memory="250GB")
time.sleep(5)
cluster.scale(10)
time.sleep(10)
client = Client(cluster)
time.sleep(5)

# declare bins and labels
labels = np.linspace(-90,89.5,359)
latbins = np.linspace(-90,90,360)
indices = np.hstack((np.arange(720,1440,1), np.arange(0,720,1)))

# rearrange the order of the frequencies, this is similar to numpy.fft.fftshift but here we repeat the Nyquist frequency
ind = list(range(720,1440))+list(range(721)) 

# declare Y bin range for eulerian calculations
Ybins = np.arange(178, 7057)
Ybins = (Ybins/(1411/36))-90

# load eulerian spectra dataset
eul_year = xr.open_dataset('/projects2/rsmas/tidaldrift/hycom_results/hycom12-spectrum-average.zarr',
                      engine = 'zarr', chunks = {'freq_time':1440})

# choose 0 m depth and filter out 'shallow' bathymetry data points
db = xr.open_dataset('/home/exf512/data/test_deep_bathym.nc')
deep_year_0m = eul_year.where(eul_year.Depth==0, drop=True).where(db.bathymetry[0,:,:])

# flip indices 
b = deep_year_0m.spectrum[indices,0,:,:]

# bin by latitude 1/2 degree and del datasets for memory
b1 = b.rename({'Latitude':'lat'})
del b
del deep_year_0m
del eul_year

# re-chunk dataset and 'groupby' latitude
b1 = b1.chunk({'Y':1000, 'X':1000})
print('b1 chunked')
a = b1.groupby_bins(group = b1.lat, bins = latbins, labels = labels)
print('a')

# average over each bin
c = a.mean()
d = c.chunk(chunks = {'freq_time':1440, 'lat_bins':5})

# open lagrangian dataset to plot indices
fdl= xr.open_dataset(
'/home/exf512/data/lag_0m_hycom/deep_fast/netcdf/deep_fast_ds_'
+str(int(i))+'.nc', chunks= {'obs':1440})

# plot and save
dt = 60*60*24
fe = (fdl.obs)*dt
f = np.append(fe, [12-1/120])
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(16, 8))
im = ax.pcolormesh(f,latbins,np.log10(lagd.transpose()),vmin=-6,vmax=0,cmap='magma')
ax.set_xlim(-4,4)
ax.set_xticks(np.arange(-4,5,1))
ax.set_ylabel('Latitude')
ax.set_yticks([-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
ax.set_ylim(-70,75)
ax.set_title('0m 2014 Average Eulerian Latitudinal Frequency Spectra')
ax.set_xlabel('Frequency (cpd)')
plt.colorbar(im,ax=ax,location='right',label=' PSD (m$^2$s$^{-1}$cpd$^{-1}$)')
#plt.savefig('/scratch/tidaldrift/test_zonal_figure_eul.png', format = 'png')

# repeat process for lagrangian
i = 1

# load lagrangian spectra
spectra = xr.open_dataset(
    '/projects2/rsmas/tidaldrift/hycom_results/hycom12-uv-lagrangian-spectrum-'
        +str(int(i))+'.zarr', engine = 'zarr')

# load ID lists for fast+deep
fast_deep_lag = xr.open_dataset(
'/home/exf512/data/lag_0m_hycom/deep_fast/netcdf/deep_fast_ds_'
+str(int(i))+'.nc', chunks= {'obs':1440})

# apply filter to lagrangian spectra dataset
df_spectra = spectra.where(
    spectra['trajectory'] == fast_deep_lag['trajectory'])

# combine datasets so freq_obs is the coordinate on latitude while flipping indices
new = xr.Dataset(data_vars = dict(lat=(['trajectory', 'freq_obs'],fast_deep_lag.lat[:,indices].data),
                       spectrum =(['trajectory', 'freq_obs'], df_spectra.spectrum[:,indices].data),
                       ), coords = dict(trajectory = (['trajectory'], fast_deep_lag.trajectory.data),
                                       freq_obs = (['freq_obs'], df_spectra.freq_obs[indices].data)))

# rechunk
new = new.chunk({'trajectory':1000, 'freq_obs':240})

# del datasets for memory
del spectra
del df_spectra
del fast_deep_lag

# bin by latitude 1/2 degree
lag = new.spectrum.groupby_bins(group = new.lat, bins = latbins, labels = labels)

# del dataset for memory
del new 
# average over each bin
lagmean = lag.mean()
print(lagmean)
print('meaned')
lagd = lagmean.chunk(chunks = {'lat_bins':5})
print('chunked')

# plot and save
dt = 60*60*24
fe = (fdl.obs)*dt
del fdl
f = np.append(fe, [12-1/120])
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(16, 8))
print('figged')
im = ax.pcolormesh(f,latbins,np.log10(lagd.transpose()),vmin=-6,vmax=0,cmap='magma')
ax.set_xlim(-4,4)
ax.set_xticks(np.arange(-4,5,1))
ax.set_ylabel('Latitude')
ax.set_yticks([-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
ax.set_ylim(-70,75)
ax.set_title('0m 2014 Average Lagrangian Latitudinal Frequency Spectra')
ax.set_xlabel('Frequency (cpd)')
plt.colorbar(im,ax=ax,location='right',label=' PSD (m$^2$s$^{-1}$cpd$^{-1}$)')
#plt.savefig('/scratch/tidaldrift/test_zonal_figure_lag.png', format = 'png')