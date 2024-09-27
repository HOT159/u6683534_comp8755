import os
#import util.data_processing_tool as dpt
from datetime import timedelta, date, datetime
# import args_parameter as args
import torch
import torchvision
import numpy as np
import random
import cv2

from torch.utils.data import Dataset,random_split
from torchvision import datasets, models, transforms

import time
import xarray as xr
from PIL import Image

def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]


class Access_AWAP_dataset(Dataset):
    def __init__(self, start_date,end_date,regin="AUS",lr_transform=None,hr_transform=None,shuffle=True):
        print("=> ACCESS_S2 & AWAP loading")
        print("=> from " + start_date.strftime("%Y/%m/%d") + " to " + end_date.strftime("%Y/%m/%d") + "")
        self.file_ACCESS_dir="/scratch/iu60/rh4668/processed_data_mask/"
        self.file_AWAP_dir="/scratch/iu60/rh4668/agcd_data_mask/"

        #self.file_ACCESS_dir="E:/VSCodeProject/Rh4668/Access_Data/"
        #self.file_AWAP_dir="E:/VSCodeProject/Rh4668/Awap_Data/"

        self.start_date=start_date
        self.end_date=end_date
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.leading_time_we_use = 7

        self.ensemble = ['e01', 'e02', 'e03', 'e04', 'e05', 'e06', 'e07', 'e08', 'e09']

        self.dates = date_range(start_date,end_date)

        self.filename_list = self.get_filename_with_time_order(self.file_ACCESS_dir)
        #self.filename_list =self.get_filename_without_en(self.file_ACCESS_dir)
        self.len=self.__len__()

        #self.filename_list = self.get_filename_with_time_order(self.file_ACCESS_dir)
        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir + "pr/daily/")
            print("no file or no permission")
        #_, _, date_for_AWAP, time_leading = self.filename_list[0]

        if shuffle:
            random.shuffle(self.filename_list)
        
    def __len__(self):
        return len(self.filename_list)
        
    def get_filename_with_no_time_order(self,rootdir):
        _files=[]
        list=os.listdir(rootdir)
        for i in range(0,len(list)):
            
            path=os.path.join(rootdir,list[i])
            if os.path.join(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files

    def get_filename_with_time_order(self, rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:
                access_path = rootdir  +"e09"+ "/da_pr_" + date.strftime("%Y%m%d") + "_" + "e09"+ ".nc"
                #                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date == self.end_date and i == 1:
                            break
                        path=[]
                        path.append(en)
                        AWAP_date = date + timedelta(i)
                        path.append(date)
                        path.append(AWAP_date)
                        path.append(i)
                        _files.append(path)

        # 最后去掉第一行，然后shuffle
        return _files

    def get_filename_without_en(self, rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            access_path = rootdir  +"e09"+ "/da_pr_" + date.strftime("%Y%m%d") + "_" + "e09" + ".nc"
                #print(access_path)
            if os.path.exists(access_path):
                for i in range(2):
                    if date == self.end_date and i == 1:
                        break
                    path=[]
                    path.append("e09")
                    AWAP_date = date + timedelta(i)
                    path.append(date)
                    path.append(AWAP_date)
                    path.append(i)
                    _files.append(path)

        # 最后去掉第一行，然后shuffle
        return _files

    def mapping(self, X, min_val=0., max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        # 将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b - a) / (Xmax - Xmin) * (X - Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t = time.time()
        lr=[]
        # read_data filemame[idx]
        en,access_date, awap_date, time_leading = self.filename_list[idx]
        lr = read_access_data(self.file_ACCESS_dir, en, access_date, time_leading, "pr")
        hr = read_awap_data(self.file_AWAP_dir, awap_date)

        return self.lr_transform(lr),self.hr_transform(hr), awap_date.strftime("%Y%m%d"), time_leading

def read_awap_data(root_dir, date_time):
    filename = root_dir + date_time.strftime("%Y-%m-%d") + ".nc"
    dataset = xr.open_dataset(filename)
    dataset = dataset.fillna(0)

        # rescale to [0,1]
    var = dataset.isel(time=0)['precip'].values
    var = (np.log1p(var)) / 7

  
    var = var[:, :,np.newaxis].astype(np.float32)  # CxLATxLON
    dataset.close()
    return var


def read_access_data(root_dir, en, date_time, leading, var_name="pr"):
    filename = root_dir + en + "/da_pr_" + date_time.strftime("%Y%m%d") + "_" + en + ".nc"

    dataset = xr.open_dataset(filename)
    dataset = dataset.fillna(0)

    # rescale to [0,1]
    var = dataset.isel(time=leading)['pr'].values 
    var = np.clip(var, 0, 1000)

    var = (np.log1p(var)) / 7

    var = cv2.resize(var, (110,86), interpolation=cv2.INTER_CUBIC)
    #var = var.transpose(1, 0)  # LATxLON
    var = var[:, :,np.newaxis].astype(np.float32)  # CxLATxLON
    dataset.close()
    return var




        

