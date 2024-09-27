import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import cv2

class L1_loss(nn.Module):
    def __init__(self):
        super(L1_loss,self).__init__()

    def forward(self,fore,obs):
        func=nn.L1Loss()
        observation=obs.squeeze()
        loss={}
        for i in range(9):
            L1=func(fore[:,i],observation)
            if i==0:
                loss[i]=L1
            else:
                loss[i]=L1+loss[i-1]
        return loss[i]/9