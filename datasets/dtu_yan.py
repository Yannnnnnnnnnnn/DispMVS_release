from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
from torchvision import transforms as T
import random
import cv2

# the DTU dataset preprocessed by Yao Yao (only for training)
class DTUDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, photo_aug=False, views_aug=False, **kwargs):
        super(DTUDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews

        self.photo_aug = photo_aug
        self.views_aug = views_aug

        assert self.mode in ["train", "val"]
        self.metas = self.build_list()
        self.color_augment = T.ColorJitter(brightness=0.5, contrast=0.5)

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # get R T
        R = extrinsics[0:3,0:3]
        T = extrinsics[0:3,3:4]
        # intrinsics: line [7-10), 3x3 matrix
        K = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        K = K * 4
        K[2,2] = 1

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[1])
        return K, R, T, depth_min, depth_max

    def read_img(self, filename):
        
        np_img = Image.open(filename)
        if self.photo_aug:
            np_img = self.color_augment(np_img)

        # scale 0~255 to 0~1
        np_img = np.array(np_img, dtype=np.float32) / 255.
        np_img = (np_img - 0.5)/0.5

        return np_img

    def read_depth(self, filename):

        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        # print('depth shape',depth.shape)
        depth = cv2.resize(depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        depth = depth[44:556,80:720]
        return depth
    
    def read_mask(self, filename):

        mask = cv2.imread(filename,0)
        mask = cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        mask = mask[44:556,80:720]
        mask = (mask>0).astype(np.float32)

        return mask

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta

        if self.views_aug:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]

        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        Ks = []
        Rs = []
        Ts = []

        for i, vid in enumerate(view_ids):

            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath, 'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename = os.path.join(self.datapath, 'Depths/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras_1/{}_train/{:0>8}_cam.txt').format(scan, vid)

            img = self.read_img(img_filename)
            img = img.transpose(2,0,1)
            imgs.append(np.expand_dims(img,axis=0))
            if i==0:
                K, R, T, depth_min,depth_max = self.read_cam_file(proj_mat_filename)
            else:
                K, R, T, _, _ = self.read_cam_file(proj_mat_filename)
                
            Ks.append(np.expand_dims(K,axis=[0,1]))
            Rs.append(np.expand_dims(R,axis=[0,1]))
            Ts.append(np.expand_dims(T,axis=[0,1]))

            if i == 0: # reference view
                mask  = self.read_mask(mask_filename)
                depth = self.read_depth(depth_filename)
            
        imgs = np.stack(imgs,axis=0)
        Ks = np.stack(Ks,axis=0)
        Rs = np.stack(Rs,axis=0)
        Ts = np.stack(Ts,axis=0)
        depth_min = np.array([depth_min]).reshape(1,1,1)
        depth_max = np.array([depth_max]).reshape(1,1,1)

        depth = depth.reshape(1,512,640)
        mask  =  mask.reshape(1,512,640)

        return imgs,Ks,Rs,Ts,depth_min,depth_max,depth,mask


# train_dataset = MVSDataset(
#     r'C:\Users\MSI-1\Desktop\part_dtu', 
#     r'../lists/dtu/train_small.txt',
#     'train',
#     3)

# imgs,Ks,Rs,Ts,depth_min,depth_max,depth,mask = train_dataset[0]

# print('imgs',imgs.shape)
# print('Ks',Ks)
# print('Rs',Rs)
# print('Ts',Ts)
# print('depth_min',depth_min)
# print('depth_min',depth_max)
# print('depth',depth.shape)
# print('mask',mask.shape,mask.dtype)






