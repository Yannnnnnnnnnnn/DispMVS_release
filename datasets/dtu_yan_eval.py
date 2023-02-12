from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *

import cv2

# the DTU dataset preprocessed by Yao Yao (only for training)
class DTU_MVSDatasetEval(Dataset):

    def __init__(self, datapath, listfile, mode, nviews, **kwargs):
        super(DTU_MVSDatasetEval, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews

        assert self.mode in ["test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)

            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    metas.append((scan, ref_view, src_views))
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

        # 1600*1200-->1600*1152
        K[1] *= (1152.0/1200.0)

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[1])

        return K, R, T, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = (np_img - 0.5)/0.5
        # 1600*1200-->1600*1152
        np_img = cv2.resize(np_img,dsize=None,fx=1.0,fy=1152.0/1200.0,interpolation=cv2.INTER_LINEAR)

        return np_img


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        Ks = []
        Rs = []
        Ts = []

        for i, vid in enumerate(view_ids):

            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,'{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams_1/{:0>8}_cam.txt'.format(scan,vid))

            img = self.read_img(img_filename)
            img = img.transpose(2,0,1)
            imgs.append(np.expand_dims(img,axis=0))
            if i==0:
                K, R, T, depth_min, depth_max = self.read_cam_file(proj_mat_filename)
            else:
                K, R, T, _, _ = self.read_cam_file(proj_mat_filename)
                
            Ks.append(np.expand_dims(K,axis=[0,1]))
            Rs.append(np.expand_dims(R,axis=[0,1]))
            Ts.append(np.expand_dims(T,axis=[0,1]))
                
        imgs = np.stack(imgs,axis=0)
        Ks = np.stack(Ks,axis=0)
        Rs = np.stack(Rs,axis=0)
        Ts = np.stack(Ts,axis=0)
        depth_min = np.array([depth_min]).reshape(1,1,1)
        depth_max = np.array([depth_max]).reshape(1,1,1)

        return imgs,Ks,Rs,Ts, depth_min, depth_max, scan+'/depth_est/'+'{:0>8}'.format(view_ids[0])

