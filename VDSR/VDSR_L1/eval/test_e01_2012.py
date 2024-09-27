import argparse
import sys
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, models, transforms
import platform
from datetime import timedelta, date, datetime
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from math import log10
import time
import cv2
import xarray as xr

sys.path.append("../")

from util.read_data import Access_AWAP_dataset
from model.vdsr import vdsr
from Loss.CRPSLoss import crps_loss,crps_loss_function
import torch
from torch.autograd import Variable
from mpl_toolkits.basemap import maskoceans

from datetime import timedelta, date, datetime

class param_args():
    '''
    Config class
    '''
    def __init__(self):
        self.train_name   ='VDSR_crps'
        self.resume     =''#module path
        self.test       =False
        self.test_model_name="VDSR_crps"
        
        
        self.n_threads    =0
        self.file_ACCESS_dir = "/scratch/iu60/rh4668/processed_data_mask"
        self.file_AWAP_dir='/scratch/iu60/rh4668/agcd_data_total_mask/'
        self.lr_size=(79,94)
        self.hr_size=(316, 376)
        
        self.precision='single'
        self.device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        
        
        self.lr           = 0.0001             # learning rate
        self.batch_size   = 8         # batch size
        self.testBatchSize= 8
        
        
        self.nEpochs      = 50            # epochs
        self.checkpoints  = './checkpoints'     # checkpoints dir
        self.seed         = 123
#         self.upscale_factor= 4
        
        self.train_start_time =date(1981,1,1)
        self.train_end_time   =date(2001,12,31)
        self.test_start_time  =date(2012,1,1)
        self.test_end_time    =date(2012,12,31)
        
        self.leading_time_we_use=7
        self.ensemble=9
        #self.domain  =[112.9, 154.25, -43.7425, -9.0]
        self.domain  = [111.975, 156.275, -44.525, -9.975]



 
def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

def get_time_without_en(rootdir,start_date,end_date):
    '''get filename first and generate label ,one different w'''
    _files = []
    dates = date_range(start_date,end_date)
    for date in dates:
        access_path = rootdir  +"e09"+ "/da_pr_" + date.strftime("%Y%m%d") + "_" + "e09" + ".nc"
                #print(access_path)
        if os.path.exists(access_path):
            _files.append(date)
    return _files

def read_access_data(path,leading):

    dataset = xr.open_dataset(path)
    dataset = dataset.fillna(0)

    var = dataset.sel(time=leading)['pr'].values 
    var = np.clip(var, 0, 1000)
    var = (np.log1p(var)) / 7

    var = cv2.resize(var, (886, 691), interpolation=cv2.INTER_CUBIC)
    #var = var.transpose(1, 0)  # LATxLON
    dataset.close()
    return var



def getitem(idx,leading):
    '''
    from filename idx get id
    return lr,hr
        '''
    # read_data filemame[idx]
    
    filename = "/scratch/iu60/rh4668/test_data/2012/e01/da_pr_" + idx.strftime("%Y%m%d") + "_e01.nc"
    var = read_access_data(filename,leading)
        
    var = var[:, :,np.newaxis].astype(np.float32)

    
    lr_transforms = transforms.Compose([transforms.ToTensor()])

    return lr_transforms(var)


def test(batch_input):
    # divide and conquer strategy due to GPU memory limit

    # _, H, W = batch_input.size()

    slice_output = model(batch_input)
    #out=torch.nn.functional.interpolate (slice_output,size=(691,886),mode='bilinear',align_corners=True)
    slice_output = slice_output.cpu().data.numpy()

    slice_output = np.clip(slice_output, 0., 1.)

    slice_output = np.squeeze(slice_output)

    #slice_output = np.clip(slice_output, 0, 1)

    return slice_output



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path=r"/scratch/iu60/rh4668/VDSR/Train/save/VDSR_L1model_epoch_9.pth"
model=torch.load(model_path,map_location=device)['model']

with torch.no_grad():
    rootdir="/scratch/iu60/rh4668/test_data/2012/e01/"
    start_date=date(2012,1,1)
    end_date=date(2012,12,31)
    files=get_time_without_en("/scratch/iu60/rh4668/test_data/2012/",start_date,end_date)
    files.sort()
    for t in files:
        fn=rootdir  +"da_pr_" + t.strftime("%Y%m%d") + "_e01.nc"
        ds_raw=xr.open_dataset(fn)
        #ds_raw = ds_raw.fillna(0)
        #ds_raw = np.clip(ds_raw, 0, 1000)
        #ds_raw = np.log1p(ds_raw) / 7
        da_selected = ds_raw.isel(time=0)["pr"]
        lon = ds_raw["lon"].values
        lat = ds_raw["lat"].values
        a = np.logical_and(lon >= 111.975, lon <= 156.275)
        b = np.logical_and(lat >= -44.525, lat <= -9.975)
        da_selected_au = da_selected[b, :][:, a].copy()

        size = (int(886), int(691))
        new_lon = np.linspace(da_selected_au.lon[0], da_selected_au.lon[-1], size[0])
        new_lon = np.float32(new_lon)
        new_lat = np.linspace(da_selected_au.lat[0], da_selected_au.lat[-1], size[1])
        new_lat = np.float32(new_lat)

        i = ds_raw['time'].values[0]

        batch_input=getitem(t,i)
        batch_input=batch_input.cuda()
        batch_output=test(batch_input)


        da_interp = xr.DataArray(np.expm1(batch_output * 7), dims=("lat", "lon"),coords={ "lat": new_lat, "lon": new_lon, "time": i}, name='pr')
        ds_total = xr.concat([da_interp], "time")

        for i in ds_raw['time'].values[1:30]:
            batch_input=getitem(t,i)
            batch_input=batch_input.cuda()

            batch_output=test(batch_input)
            da_interp = xr.DataArray(np.expm1(batch_output * 7), dims=("lat", "lon"),coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
            expanded_da=xr.concat([da_interp],"time")
            ds_total=xr.merge([ds_total,expanded_da])

        save_path="/scratch/iu60/rh4668/VDSR/eval/2012/e01/" + t.strftime("%Y%m%d")  + ".nc"
        ds_total.to_netcdf(save_path)
