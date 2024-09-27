import xarray as xr
import matplotlib
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import cartopy.crs as ccrs
import os
import cv2

from netCDF4 import Dataset, num2date, date2num
#from libtiff import TIFF
from datetime import timedelta, date
import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.basemap import Basemap,maskoceans
#from tqdm import tqdm

levels = {}
#levels["crps"]   = [0,0.2,0.4,0.6,0.8,1.0] 
levels["crpsss"]   = [-0.8,-0.4,-0.2,0,0.2,0.4,0.8] 
levels["new"]   = [0, 0.1, 1.0 ,5.0, 10.0, 20.0, 30.0, 40.0, 60.0 ,100, 150] 
levels["mae"]   = [0, 0.5, 1 ,1.5, 2, 2.5, 3, 4, 6 ,8, 10] 
levels["hour"]  = [0., 0.2, 1, 5,  10,  20,  30,   40,   60,   80,  100,  150]
levels["day"]   = [0., 0.2, 5, 10,  20,  30,  40,  60,  100,  150,  200,  300]
levels["week"]  = [0., 0.2, 10,  20,  30,  50, 100,  150,  200,  300,  500, 1000]
levels["month"] = [0., 10, 20,  30,  40,  50, 100,  200,  300,  500, 1000, 1500]
levels["year"]  = [0., 50, 100, 200, 300, 400, 600, 1000, 1500, 2000, 3000, 5000]
enum={0:"0600",1:"1200",2:"1800",3:"0000",4:"0600"}

prcp_colours = [
                   "#FFFFFF", 
                   '#edf8b1',
                   '#c7e9b4',
                   '#7fcdbb',
                   '#41b6c4',
                   '#1d91c0',
                   '#225ea8',
                   '#253494',
                   '#4B0082',
                   "#800080",
                   '#8B0000']

prcp_colormap = matplotlib.colors.ListedColormap(prcp_colours)

def draw_aus(var,lat,lon,domain = [111.975, 156.275, -44.525, -9.975], mode="pr" , titles_on = True, title = "CRPS of precipation in 2012", colormap = prcp_colormap, cmap_label = "CRPS-Skill Score",save=False,path=""):
    """ basema_ploting .py
This function takes a 2D data set of a variable from AWAP and maps the data on miller projection. 
The map default span is longitude between 111.975E and 156.275E, and the span for latitudes is -44.525 to -9.975. 
The colour scale is YlGnBu at 11 levels. 
The levels specifed are suitable for annual rainfall totals for Australia. 
"""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.basemap import Basemap,maskoceans
    
    if mode == "pr":
        level = 'new'
    
    # crps-ss
    if mode == "crps-ss":
        level = "crpsss"
             
    if mode == "mae":
        level = "mae"  
    
    fig=plt.figure()
    level=levels[level]
    map = Basemap(projection = "mill", llcrnrlon = domain[0], llcrnrlat = domain[2], urcrnrlon = domain[1], urcrnrlat = domain[3], resolution = 'l')
    map.drawcoastlines()
    #map.drawmapboundary()
    #map.drawparallels(np.arange(-90., 120., 5.),labels=[1,0,0,0])
    #map.drawmeridians(np.arange(-180.,180., 5.),labels=[0,0,0,1])
    llons, llats = np.meshgrid(lon, lat) # 将维度按照 x,y 横向竖向
   # print(lon.shape,llons.shape)
    x,y = map(llons,llats)
   # print(x.shape,y.shape)
    
    norm = BoundaryNorm(level, len(level)-1)
    
    # red square
    #var[255:260,205:510]= 1000
    #var[495:500,210:510]= 1000
    #var[260:500,205:210]= 1000
    #var[260:500,505:510]= 1000
    
    data=xr.DataArray(var,coords=[lat,lon],dims=["lat","lon"])
    
    # pr
    if mode == "pr":
        cs = map.pcolormesh(x, y, data, norm = norm, cmap = colormap) 
    
    # crps-ss
    if mode == "crps-ss":
        cs = map.pcolormesh(x, y, data, cmap="RdBu",vmin=-0.8,vmax=0.8) 
        
    if mode == "mae":
        cs = map.pcolormesh(x, y, data, norm = norm, cmap = colormap) 
    
    if titles_on:
        # label with title, latitude, longitude, and colormap
        
        plt.title(title)
        #plt.xlabel("\n\nLongitude")
        #plt.ylabel("Latitude\n\n")
        
        # color bar
        cbar = plt.colorbar(ticks = level[:-1], shrink = 0.8, extend = "max")#shrink = 0.8
        cbar.ax.set_ylabel(cmap_label)
        
        #cbar.ax.set_xticklabels(level) #报错
    
    # plt.plot([-1000,1000],[900,1000], c="b", linewidth=2, linestyle=':')
    
    if save:
        plt.savefig(path)
    else:
        plt.show()
    plt.cla()
    plt.close("all")
    return