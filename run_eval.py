
import cv2
import torch
import numpy as np
from torch import optim

import eval
from model import DispMVS
from datasets.dtu_yan import MVSDataset

device = 'cuda'
epoch = 50
flow_iter = 12
learning_rate = 0.0001

test_dataset = MVSDataset(
    r'/data/yqs/dtu_training', 
    r'./lists/dtu/val.txt',
    'val',
    3)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

train_dir = './checkpoint_large/'
model_dir = './checkpoint_large/models/'

def viz_depth(path_name,depth,mask):

    depth_viz = depth[0,0,:,:].detach().cpu().numpy()
    depth_min = 300
    depth_max = 1200
    depth_rng = (depth_max-depth_min)
    depth = depth.clip(depth_min,depth_max)
    depth_viz = 255*(depth_viz-depth_min)/depth_rng
    depth_viz = depth_viz*mask[0,0].detach().cpu().numpy()
    depth_viz = depth_viz.astype(np.uint8)
    cv2.imwrite(path_name,cv2.applyColorMap(depth_viz,cv2.COLORMAP_JET))

def viz_diff(path_name,depth,mask):

    depth_viz = depth[0,0,:,:].detach().cpu().numpy()
    depth = depth.clip(0,30)
    
    depth_viz = depth_viz*mask[0,0].detach().cpu().numpy()

    depth_viz = depth_viz.astype(np.uint8)
    cv2.imwrite(path_name,cv2.applyColorMap(depth_viz,cv2.COLORMAP_JET))


model = DispMVS()
model.to(device)
model.load_state_dict(torch.load('./checkpoint_large/models/16_0.031843187732954636.pt'))

model.eval()
with torch.no_grad():
    abs_err_mean = 0
    acc_1mm_mean = 0
    acc_2mm_mean = 0
    acc_4mm_mean = 0
    acc_cnt = 0
    for i, (imgs,Ks,Rs,Ts,depth_min,depth,mask) in enumerate(test_loader):
    
        imgs = imgs.to(device)
        Ks = Ks.to(device)
        Rs = Rs.to(device)
        Ts = Ts.to(device)
        depth_min = depth_min.float().to(device)
        mask = mask.to(device)
        depth = depth.to(device)

        depth_iter = model(imgs,Ks,Rs,Ts,flow_iter,depth_min,depth_min*2)

        abs_err,acc_1mm,acc_2mm,acc_4mm = eval.eval(depth.cpu().numpy(),mask.cpu().numpy(),depth_iter[flow_iter-1].cpu().numpy())
        print('eval: ',i,"in",len(test_loader),":",abs_err,acc_1mm,acc_2mm,acc_4mm)

        abs_err_mean += abs_err
        acc_1mm_mean += acc_1mm
        acc_2mm_mean += acc_2mm
        acc_4mm_mean += acc_4mm
        acc_cnt += 1

        if acc_cnt%5==0:
            cv2.imwrite(train_dir+'val_ref_img.png',(imgs[0,0,0,:,:,:].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            viz_depth(train_dir+'val_depth_gt.png',depth,mask)
            viz_depth(train_dir+'val_depth_pr.png',depth_iter[flow_iter-1],mask)
            viz_diff(train_dir+'val_depth_diff.png',(depth_iter[flow_iter-1]-depth).abs(),mask)

        if acc_cnt==2:
            break
