import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import cv2

class L2_loss(nn.Module):
    def __init__(self):
        super(L2_loss,self).__init__()

    def forward(self,fore,obs):
        func=nn.MSELoss(reduction='mean')
        observation=obs.squeeze()
        loss={}
        for i in range(9):
            L2=func(fore[:,i],observation)
            if i==0:
                loss[i]=L2
            else:
                loss[i]=L2+loss[i-1]
        return loss[i]/9