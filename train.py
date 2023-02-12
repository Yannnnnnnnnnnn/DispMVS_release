import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import os
import gc
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from utils import *
from eval import eval
from loss import inverse_loss
from model import DispMVS
from datasets.dtu_yan import DTUDataset
from datasets.blended_yan import BlendedDataset

parse = argparse.ArgumentParser()
parse.add_argument('--logdir',type=str,required=True)
parse.add_argument('--dataset',type=str,required=True)
parse.add_argument('--data_root',type=str,required=True)
parse.add_argument('--train_list',type=str,required=True)
parse.add_argument('--valid_list',type=str,required=True)
parse.add_argument('--lr',type=float,default=0.0001)
parse.add_argument('--bs',type=int,default=1)
parse.add_argument('--epoch',type=int,default=100)
parse.add_argument('--show_frequent',type=int,default=50)
parse.add_argument('--checkpoint',type=str,default=None)
parse.add_argument('--dropout', type=float, default=0.0)
parse.add_argument('--iter', nargs="+", type=int, default=[2, 8], help='number of iteration')
parse.add_argument('--views', type=int, default=3)
parse.add_argument('--photo_aug', dest='photo_aug', action='store_true') 
parse.add_argument('--views_aug', dest='views_aug', action='store_true')
parse.add_argument('--clip', type=float, default=1.0)
parse.add_argument('--device',default='cuda')
args = parse.parse_args()

print('photo-aug',args.photo_aug)
print('views-aug',args.views_aug)

device = args.device
batch_size = args.bs
epoch = args.epoch
learning_rate = args.lr
show_frequent = args.show_frequent
train_dir = args.logdir
model_dir = os.path.join(args.logdir,'models')

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if args.dataset == 'DTU':

    train_dataset = DTUDataset(
        args.data_root, 
        args.train_list,
        'train',
        args.views,
        args.photo_aug,
        args.views_aug)

    val_dataset = DTUDataset(
        args.data_root,
        args.valid_list,
        'val',
        args.views,
        False,
        False)
        
elif args.dataset == 'Blended':
    train_dataset = BlendedDataset(
        args.data_root, 
        args.train_list,
        'train',
        args.views,
        args.photo_aug,
        args.views_aug)

    val_dataset = BlendedDataset(
        args.data_root,
        args.valid_list,
        'val',
        args.views,
        False,
        False)    
else:
    print("Wrong dataset type!")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False)

model = DispMVS(args)
model.to(device)

if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[10, 12, 14], gamma=0.5)
eval_log = open(train_dir+'/log.txt','w')

for e in range(epoch):

    model.train()
    train_loss = 0
    train_cnt = 0
    for imgs,Ks,Rs,Ts,depth_min,depth_max,depth,mask in tqdm(train_loader):
        
        imgs = imgs.to(device)
        Ks = Ks.to(device)
        Rs = Rs.to(device)
        Ts = Ts.to(device)
        depth_min = 1.0/depth_min.float().to(device)
        depth_max = 1.0/depth_max.float().to(device)
        mask = mask.to(device)
        depth = depth.to(device)

        b,_,h,w = depth.shape
        init_depth = torch.rand((b,1,h//16,w//16),device=device)*(depth_min-depth_max) + depth_max
        init_depth = 1.0/init_depth

        depths_pred = model(imgs,Ks,Rs,Ts,args.iter,init_depth,depth_min,depth_max)
        
        # clip grad to avoid nan
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        loss = inverse_loss(depths_pred,depth,mask,depth_min,depth_max)        
        
        # directly removing nan
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_cnt += 1
        train_loss += loss.detach().cpu().numpy()

        if train_cnt%show_frequent==0:
            print('loss: ',loss.detach().cpu().numpy())

            cv2.imwrite(train_dir+'/tra_ref_img.png',(imgs[0,0,0,:,:,:].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            depth_min,depth_max = viz_depth(train_dir+'/tra_depth_gt.png',depth,mask)
            viz_depth(train_dir+'/tra_depth_pr.png',depths_pred[0][-1],mask,depth_min,depth_max)
            viz_diff(train_dir+'/tra_depth_diff.png',(depths_pred[0][-1]-depth).abs(),mask)
        
  
    gc.collect()

    model.eval()
    with torch.no_grad():

        abs_err_mean = 0
        acc_1mm_mean = 0
        acc_2mm_mean = 0
        acc_4mm_mean = 0
        mask_mean = 0
        acc_cnt = 0
        
        for imgs,Ks,Rs,Ts,depth_min,depth_max,depth,mask in tqdm(val_loader):
        
            imgs = imgs.to(device)
            Ks = Ks.to(device)
            Rs = Rs.to(device)
            Ts = Ts.to(device)
            depth_min = 1.0/depth_min.float().to(device)
            depth_max = 1.0/depth_max.float().to(device)
            mask = mask.to(device)
            depth = depth.to(device)

            b,_,h,w = depth.shape
            init_depth = torch.rand((b,1,h//16,w//16),device=device)*(depth_min-depth_max) + depth_max
            init_depth = 1.0/init_depth

            depths_pred = model(imgs,Ks,Rs,Ts,args.iter,init_depth,depth_min,depth_max)

            abs_err,acc_1mm,acc_2mm,acc_4mm,mask_num = eval(depth,mask>0,depths_pred[-1])

            abs_err_mean += abs_err.cpu().numpy()
            acc_1mm_mean += acc_1mm.cpu().numpy()
            acc_2mm_mean += acc_2mm.cpu().numpy()
            acc_4mm_mean += acc_4mm.cpu().numpy()
            mask_mean    += mask_num.cpu().numpy()

            acc_cnt += 1
            if acc_cnt%show_frequent==0:
                print('eval: ',
                    (abs_err_mean/mask_mean),
                    (acc_1mm_mean/mask_mean),
                    (acc_2mm_mean/mask_mean),
                    (acc_4mm_mean/mask_mean))
                
                cv2.imwrite(train_dir+'/val_ref_img.png',(imgs[0,0,0,:,:,:].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                depth_min,depth_max = viz_depth(train_dir+'/val_depth_gt.png',depth,mask)
                viz_depth(train_dir+'/val_depth_pr.png',depths_pred[-1],mask,depth_min,depth_max)
                viz_diff(train_dir+'/val_depth_diff.png',(depths_pred[-1]-depth).abs(),mask)

        eval_log.write(str(e)+'\t'+str(abs_err_mean/mask_mean)+'\t'+str(acc_1mm_mean/mask_mean)+'\t'+str(acc_2mm_mean/mask_mean)+'\t'+str(acc_4mm_mean/mask_mean)+'\n')
        eval_log.flush()

    gc.collect()

    scheduler.step()
    eval_log.write('lr:'+str(scheduler.get_last_lr())+'\n')
    eval_log.flush()

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(),model_dir+'/'+str(e)+'_'+str(train_loss/train_cnt)+'.pt')
    else:
        torch.save(model.state_dict(),model_dir+'/'+str(e)+'_'+str(train_loss/train_cnt)+'.pt')
