import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import cv2

from properscoring import crps_ensemble, crps_quadrature, crps_gaussian

class crps_loss(nn.Module):
    def __init__(self):
        super(crps_loss,self).__init__()
    
    def forward(self,fore,obs):

        shape_forecast=fore.shape
        score={}
        for i in range(shape_forecast[0]):
            forecasts=fore[i]
            observations=obs[i]

            forecasts=forecasts.squeeze()
            forecasts=forecasts.permute(1,2,0)
            observations=observations.squeeze()
            if observations.ndim == forecasts.ndim - 1:
                assert observations.shape==forecasts.shape[:-1]
                observations=observations.unsqueeze(-1)
                score[i]=torch.mean(torch.abs(forecasts-observations),dim=-1)
                forecasts_diff=(forecasts.unsqueeze(-1)-forecasts.unsqueeze(-2))
            if i==0:
                score[i]=torch.mean(score[i]+ (-0.5 * torch.mean(torch.abs(forecasts_diff),
                                               dim=(-2, -1))) )
            else:
                score[i]=torch.mean(score[i]+ (-0.5 * torch.mean(torch.abs(forecasts_diff),
                                               dim=(-2, -1))) )+score[i-1]

            
        return score[i]/shape_forecast[0]


class crps_loss_function(nn.Module):
        def __init__(self):
            super(crps_loss_function,self).__init__()
        def forward(self,fore,obs):
            o=obs.squeeze()
            ob=o.cpu().detach().numpy()
            #observation=ob[:, :,np.newaxis].astype(np.float32)
            observation=ob
            var=fore.cpu().detach().numpy().squeeze()
            forester = cv2.resize(np.transpose(var,(1,2,0)), (886,691), interpolation=cv2.INTER_CUBIC)
            crps_score=crps_ensemble(observation,forester)
            return torch.tensor(np.mean(crps_score))





    
