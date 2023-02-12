import torch
import torch.nn.functional as F

def inverse_loss(depths,gt,mask,depth_min,depth_max):

    loss_iter = 0
    gamma = 0.9

    # B,1,H,W
    weight = 1.0

    gt = 1.0/(gt+1e-6)
    gt = (gt-depth_max)/(depth_min-depth_max)
    flow_iter = len(depths[0])
    for i in range(flow_iter):

        depth = 1.0/(depths[0][i]+1e-6)
        depth = (depth-depth_max)/(depth_min-depth_max)

        diff = ( depth - gt ).abs()
        loss_temp = diff[mask>0].mean()

        loss_iter += weight*loss_temp
        weight *= gamma

    # B,1,H/4,W/4
    weight = 1.0
    gt = F.interpolate(gt,scale_factor=0.25,recompute_scale_factor=True)
    mask = F.interpolate(mask,scale_factor=0.25,recompute_scale_factor=True)
    flow_iter = len(depths[1])
    for i in range(flow_iter):

        depth = 1.0/(depths[1][i]+1e-6)
        depth = (depth-depth_max)/(depth_min-depth_max)

        diff = ( depth - gt ).abs()
        loss_temp = diff[mask>0].mean()

        loss_iter += weight*loss_temp
        weight *= gamma
      
    return loss_iter