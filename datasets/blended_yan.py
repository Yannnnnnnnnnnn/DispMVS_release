from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
from torchvision import transforms as T
import random
import cv2

# the DTU dataset preprocessed by Yao Yao (only for training)
class BlendedDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, photo_aug=False, views_aug=False, **kwargs):
        super(BlendedDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews

        self.photo_aug = photo_aug
        self.views_aug = views_aug

        assert self.mode in ['train', 'val', 'all']
        self.metas = self.build_list()

        self.color_augment = T.ColorJitter(brightness=0.5, contrast=0.5)

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            self.scans = [line.rstrip() for line in f.readlines()]
        for scan in self.scans:
            with open(os.path.join(self.datapath, scan, "cams/pair.txt")) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    nei_view = f.readline().rstrip().split()
                                        
                    # ignore not enough pair
                    if float(nei_view[0]) < self.nviews:
                        continue
                    
                    src_views = [int(x) for x in nei_view[1::2]]
                    metas += [(scan, ref_view, src_views)]
                    
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, scan, filename):
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
        # (576,768) -> (512,640)
        # K[0,2] -= 64
        # K[1,2] -= 32

        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])
        
        return K, R, T, depth_min, depth_max

    def read_img(self, filename):
        np_img = Image.open(filename)
        if self.photo_aug:
            np_img = self.color_augment(np_img)
        # scale 0~255 to 0~1
        np_img = np.array(np_img, dtype=np.float32) / 255.
        np_img = (np_img - 0.5)/0.5
        # np_img = cv2.resize(np_img, (640, 512), interpolation=cv2.INTER_LINEAR)
        # np_img = np_img[32:-32,64:-64,:] 

        return np_img

    def read_depth_mask(self, scan, filename, depth_min, depth_max):
        
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        # depth = cv2.resize(depth, (640, 512), interpolation=cv2.INTER_NEAREST)
        # depth = depth[32:-32,64:-64] 

        mask = (depth>=depth_min) & (depth<=depth_max)
        mask = mask.astype(np.float32)

        # clip outlier
        depth = depth.clip(depth_min,depth_max)

        return depth, mask

    def __getitem__(self, idx):

        meta = self.metas[idx]
        scan, ref_view, src_views = meta

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

            img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            img = img.transpose(2,0,1)
            
            if i == 0: # reference view
                K, R, T, depth_min,depth_max = self.read_cam_file(scan, proj_mat_filename)
                depth,mask = self.read_depth_mask(scan, depth_filename, depth_min, depth_max)
            else:
                K, R, T, _, _ = self.read_cam_file(scan, proj_mat_filename)

            imgs.append(np.expand_dims(img,axis=0))
            Ks.append(np.expand_dims(K,axis=[0,1]))
            Rs.append(np.expand_dims(R,axis=[0,1]))
            Ts.append(np.expand_dims(T,axis=[0,1]))

        imgs = np.stack(imgs,axis=0)
        Ks = np.stack(Ks,axis=0)
        Rs = np.stack(Rs,axis=0)
        Ts = np.stack(Ts,axis=0)
        depth_min = np.array([depth_min]).reshape(1,1,1)
        depth_max = np.array([depth_max]).reshape(1,1,1)

        depth = depth.reshape(1,576,768)
        mask  = mask.reshape(1,576,768)        
        
        return imgs,Ks,Rs,Ts,depth_min,depth_max,depth,mask








