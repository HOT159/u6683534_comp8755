
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

sys.path.append("../")

#read data package
from util.read_data import Access_AWAP_dataset
from model.RFDN import RFDN,RFDNX4X2
#导入loss_function
from Loss.CRPSLoss import crps_loss,crps_loss_function
import torch
from datetime import timedelta, date, datetime

class param_args():
    '''
    Config class
    '''
    def __init__(self):
        self.train_name   ='pcfsr_pr_crps_only'
        self.resume     =''#module path
        self.test       =False
        self.test_model_name="pcfsr"
        
        
        self.n_threads    =0
        self.file_ACCESS_dir = "/scratch/iu60/rh4668/processed_data"
        self.file_AWAP_dir='/scratch/iu60/rh4668/agcd_data_total/'
        self.lr_size=(79,94)
        self.hr_size=(316, 376)
        
        self.precision='single'
        self.device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        
        
        self.lr           = 0.001             # learning rate
        self.batch_size   = 8         # batch size
        self.testBatchSize= 8
        
        
        self.nEpochs      = 100            # epochs
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

def write_log(log,args):
    print(log)
    if not os.path.exists("./save/"+args.train_name+"/"):
        os.makedirs("./save/"+args.train_name+"/")
    my_log_file=open("./save/"+args.train_name + '/train.txt', 'a')
#     log="Train for batch %d,data loading time cost %f s"%(batch,start-time.time())
    my_log_file.write(log + '\n')
    my_log_file.close()
    return

class trainer():
    def __init__(self):
        self.args=param_args()
        
    def main(self):
        for i in self.args.__dict__:
            write_log((i.ljust(20)+':'+str(self.args.__dict__[i])),self.args)

        lr_transforms = transforms.Compose([
            transforms.ToTensor()          
        ])
        hr_transforms = transforms.Compose([
            transforms.ToTensor()          
        ])

        train_data=Access_AWAP_dataset(self.args.train_start_time,self.args.train_end_time,lr_transform=lr_transforms,hr_transform=hr_transforms)
        val_data=Access_AWAP_dataset(self.args.test_start_time,self.args.test_end_time,lr_transform=lr_transforms,hr_transform=hr_transforms)
        train_dataloders =DataLoader(train_data,batch_size=self.args.batch_size,shuffle=True,num_workers=self.args.n_threads)
        val_dataloders =DataLoader(val_data,batch_size=self.args.batch_size,shuffle=True,num_workers=self.args.n_threads)

        model=RFDN()
        model.to(self.args.device)
        criterion=crps_loss()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        if torch.cuda.device_count() > 1:
            write_log("!!!!!!!!!!!!!Let's use"+str(torch.cuda.device_count())+"GPUs!",self.args)
            model = nn.DataParallel(model,range(torch.cuda.device_count()))
        else:
            write_log("Let's use"+str(torch.cuda.device_count())+"GPUs!",self.args)
        
        start_epoch=0
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                write_log("=> loading checkpoint '{}'".format(self.args.resume),self.args)
                checkpoint = torch.load(self.args.resume)
                start_epoch = checkpoint["epoch"] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                model.load_state_dict(checkpoint["model"].state_dict())
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99,last_epoch=start_epoch)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))  
        

        write_log("start",self.args)
        max_error=np.inf
        best_test=np.inf
        train_loss_list=[]
        test_loss_list=[]
        for epoch in range(start_epoch, self.args.nEpochs + 1):
            start=time.time()
            write_log("epoch = "+str(epoch)+", lr = "+str(optimizer.param_groups[0]["lr"]),self.args)
            model.train()    
            avg_loss=0

            for iteration, (pr,hr,_,_) in enumerate(train_dataloders):
                pr,hr= self.prepare([pr,hr],self.args.device)
                out_ensemble = model(pr)
                out=torch.nn.functional.interpolate (out_ensemble,size=(691,886),mode='bicubic', align_corners=True)
                #print(out.shape)
                loss = criterion(out,hr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss+=loss.item()
            write_log("epoche: %d,lr: %f,time cost %f s, train_loss: %f "%(
                              epoch,
                              optimizer.state_dict()['param_groups'][0]['lr'],
                              time.time()-start,
                              avg_loss / len(train_dataloders),
                         ),self.args)
            scheduler.step()

            start=time.time()
            model.eval()

            with torch.no_grad():
                avg_crps=0
                for iteration,(pr,hr,_,_) in enumerate(val_dataloders):
                    pr,hr=self.prepare([pr,hr],self.args.device)
                    out_ensemble=model(pr)
                    out=torch.nn.functional.interpolate (out_ensemble,size=(691,886),mode='bicubic',align_corners=True)
                    crps_score=criterion(out, hr)
                    avg_crps+=crps_score.item()

                test_crps=avg_crps/len(val_dataloders)
                write_log("epoche: %d,lr: %f,time cost %f s, test: %f "%(
                          epoch,
                          optimizer.state_dict()['param_groups'][0]['lr'],
                          time.time()-start,
                          test_crps,
                     ),self.args)
            
            if(best_test>=test_crps):
                best_test=test_crps
                self.save_checkpoint(model,epoch,optimizer)
                write_log("best crps epoches: %d: best_test: %f" %(epoch, best_test),self.args)

    def prepare(self,l,device=False):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            if self.args.precision == 'single': tensor = tensor.float()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]    

    def save_checkpoint(self,model, epoch,optimizer):
        model_folder = "./save/"+self.args.train_name
        model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
        state = {"epoch": epoch ,
                 "model": model ,
                 'optimizer': optimizer.state_dict(),
                 'argparse': self.args
                }
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

if __name__=='__main__':
    trainner=trainer()
    trainner.main()    