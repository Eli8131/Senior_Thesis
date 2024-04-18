import numpy as np
import xarray as xr
import zarr
import sys
import time
import scipy as sc

lon = np.linspace(-180,180, 360*2)
lat = np.linspace(-90, 90, 180*2)

for i in np.arange(1,12):

    # open filtered time series
    zp = xr.open_dataset('/scratch/tidaldrift/filtered_wavelet/LF/hycom_LF_15m_'+str(int(i))+'_zp.zarr',engine='zarr')["zp"]
    zn = xr.open_dataset('/scratch/tidaldrift/filtered_wavelet/LF/hycom_LF_15m_'+str(int(i))+'_zn.zarr',engine='zarr')["zn"]

    # open corresponding lat and lon
    ids = np.loadtxt('/home/exf512/data/lag_15m_hycom/deep_fast_IDs_15m/deep_fast_IDs_15m'+str(int(i))).astype(int)
    lat_t = xr.open_dataset('/projectnb/msldrift/pra-drifters/hycom/global_hycom_15m_step_'+str(int(i))+'_withmean_chunked.zarr', 
                            engine = 'zarr')["lat"][ids,:]
    lon_t = xr.open_dataset('/projectnb/msldrift/pra-drifters/hycom/global_hycom_15m_step_'+str(int(i))+'_withmean_chunked.zarr', 
                            engine = 'zarr')["lon"][ids,:]

    zp_real_mean = sc.stats.binned_statistic_2d(lon_t.to_numpy().flatten(), 
                                   lat_t.to_numpy().flatten(),
                                   np.real(zp).to_numpy().flatten(),
                                   statistic = np.mean,
                                   bins= [lon,lat])

    zp_imag_mean = sc.stats.binned_statistic_2d(lon_t.to_numpy().flatten(), 
                                   lat_t.to_numpy().flatten(),
                                   np.imag(zp).to_numpy().flatten(),
                                   statistic = np.mean,
                                   bins= [lon,lat])

    zp_var = sc.stats.binned_statistic_2d(lon_t.to_numpy().flatten(), 
                                   lat_t.to_numpy().flatten(),
                                   zp.to_numpy().flatten(),
                                   statistic = np.var,
                                   bins= [lon,lat])

    dsp = xr.Dataset(data_vars=dict(zp_real_mean = (['lon','lat'], zp_real_mean.statistic.real),
                                     zp_imag_mean = (['lon','lat'], zp_imag_mean.statistic.real),
                                     zp_var = (['lon','lat'], zp_var.statistic.real),),
              coords = dict(lon = zp_real_mean.x_edge[0:-1], lat = zp_real_mean.y_edge[0:-1]),)
    
    dsp.to_netcdf('/projectnb/msldrift/tidaldrift/faigle/wavelet_stats/LF/hycom_LF_15m_stats_step'+str(int(i))+'_cc_withmean.nc')

    zn_real_mean = sc.stats.binned_statistic_2d(lon_t.to_numpy().flatten(), 
                                   lat_t.to_numpy().flatten(),
                                   np.real(zn).to_numpy().flatten(),
                                   statistic = np.mean,
                                   bins= [lon,lat])

    zn_imag_mean = sc.stats.binned_statistic_2d(lon_t.to_numpy().flatten(), 
                                   lat_t.to_numpy().flatten(),
                                   np.imag(zn).to_numpy().flatten(),
                                   statistic = np.mean,
                                   bins= [lon,lat])

    zn_var = sc.stats.binned_statistic_2d(lon_t.to_numpy().flatten(), 
                                   lat_t.to_numpy().flatten(),
                                   zn.to_numpy().flatten(),
                                   statistic = np.var,
                                   bins= [lon,lat])
    

    dsn = xr.Dataset(data_vars=dict(zn_real_mean = (['lon','lat'], zn_real_mean.statistic.real),
                                     zn_imag_mean = (['lon','lat'], zn_imag_mean.statistic.real),
                                     zn_var = (['lon','lat'], zn_var.statistic.real),),
              coords = dict(lon = zn_real_mean.x_edge[0:-1], lat = zn_real_mean.y_edge[0:-1]),)
    
    dsn.to_netcdf('/projectnb/msldrift/tidaldrift/faigle/wavelet_stats/LF/hycom_LF_15m_stats_step'+str(int(i))+'_cw_withmean.nc')
    
    print('step '+str(i)+' done!')
    
