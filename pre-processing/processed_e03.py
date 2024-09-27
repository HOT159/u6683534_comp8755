import xarray as xr
import matplotlib
# from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.basemap import maskoceans
# import cartopy.crs as ccrs
import os

for file in (os.listdir("/g/data/ux62/access-s2/hindcast/raw_model/atmos/pr/daily/e03/")):
    ds_raw = xr.open_dataset("/g/data/ux62/access-s2/hindcast/raw_model/atmos/pr/daily/e03/" + file) * 86400
    ds_raw = ds_raw.fillna(0)
    da_selected = ds_raw.isel(time=0)['pr']

    lon = ds_raw["lon"].values
    lat = ds_raw["lat"].values
    a = np.logical_and(lon >= 111.975, lon <= 156.275)
    b = np.logical_and(lat >= -44.525, lat <= -9.975)

    da_selected_au = da_selected[b, :][:, a].copy()

    # resize lat & lon
    n = 1.5  # 60km -> 40km   == 1.5 scale

    size = (int(da_selected_au.lon.size * n),
            int(da_selected_au.lat.size * n))
    new_lon = np.linspace(
        da_selected_au.lon[0], da_selected_au.lon[-1], size[0])
    new_lon = np.float32(new_lon)
    new_lat = np.linspace(da_selected_au.lat[0], da_selected_au.lat[-1], size[1])
    new_lat = np.float32(new_lat)
    lons, lats = np.meshgrid(new_lon, new_lat)
    # interp and merge
    i = ds_raw['time'].values[0]
    da_selected = ds_raw.sel(time=i)['pr']
    da_selected_au = da_selected[b, :][:, a].copy()
    temp = cv2.resize(da_selected_au.values, size,interpolation=cv2.INTER_CUBIC)
    temp = np.clip(temp, 0, None)
    #mask
    temp= maskoceans(lons, lats, temp)

    da_interp = xr.DataArray(temp, dims=("lat", "lon"),coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
    ds_total = xr.concat([da_interp], "time")

    for i in ds_raw['time'].values[:7]:
        ds_selected_domained = ds_raw.sel(time=i)['pr']
        da_selected_au = ds_selected_domained[b, :][:, a].copy()
        temp = cv2.resize(da_selected_au.values, size,interpolation=cv2.INTER_CUBIC)
        temp = np.clip(temp, 0, None)
        temp= maskoceans(lons, lats, temp)
        da_interp = xr.DataArray(temp, dims=("lat", "lon"),coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
        expanded_da = xr.concat([da_interp], "time")
        ds_total = xr.merge([ds_total, expanded_da])

    # save to netcdf
    ds_total.to_netcdf("/scratch/iu60/rh4668/processed_data_mask/e03/" + file)
    ds_raw.close()
