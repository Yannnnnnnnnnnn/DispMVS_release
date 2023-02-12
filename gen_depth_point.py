import argparse
import os
from matplotlib.pyplot import sca
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from model import *
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
from datasets.dtu_yan_eval import DTU_MVSDatasetEval
from datasets.tankstemple_yan_eval import TanksTemple_MVSDatasetEval
import utils 

cudnn.benchmark = True

def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")

# view fuse; pixel diff; relative depth diff;
tankstemple_hyper = {
    'Family':    [5,2.0, 0.005],
    'Francis':   [8,2.0,0.0025],
    'Horse':     [4,1.0,  0.01],
    'Lighthouse':[7,3.0, 0.005],
    'M60':       [5,2.0,0.0025],
    'Panther':   [6,4.0, 0.005],
    'Playground':[7,3.0, 0.005],
    'Train':     [6,2.5, 0.005],
    'Auditorium':[3,1.0,  0.01],
    'Ballroom':  [3,2.5,  0.01],
    'Courtroom': [4,1.0,  0.01],
    'Museum':    [4,1.0,  0.01],
    'Palace':    [4,2.5,  0.01],
    'Temple':    [3,2.5,  0.01]
}

dtu_hyper = {
    'scan1':   [3, 0.5,0.01],
    'scan4':   [3, 0.5,0.01],
    'scan9':   [3,0.25,0.01],
    'scan10':  [3,0.25,0.01],
    'scan11':  [3, 0.5,0.01],
    'scan12':  [3,0.25,0.01],
    'scan13':  [3,0.75,0.01],
    'scan15':  [3,0.25,0.01],
    'scan23':  [3,0.25,0.01],
    'scan24':  [3, 0.5,0.01],
    'scan29':  [3, 0.5,0.01],
    'scan32':  [3,0.25,0.01],
    'scan33':  [3, 0.5,0.01],
    'scan34':  [3,0.25,0.01],
    'scan48':  [3, 0.5,0.01],
    'scan49':  [3,0.25,0.01],
    'scan62':  [3, 0.5,0.01],
    'scan75':  [3,0.25,0.01],
    'scan77':  [3,0.25,0.01],
    'scan110': [3,0.25,0.01],
    'scan114': [3, 0.5,0.01],
    'scan118': [3,0.75,0.01]
}


parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--dataset', default='dtu', help='select dataset')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--testlist', help='testing scan list')
parser.add_argument('--number_views_pred', type=int, default=3, help='number of views used to estimate depth map')
parser.add_argument('--iter', nargs="+", type=int, default=[10, 2], help='number of iteration')
parser.add_argument('--abs_pixel_diff', type=float, default=1, help='pixel difference threshold')
parser.add_argument('--relative_depth_diff', type=float, default=0.01, help='relative depth difference threshold')
parser.add_argument('--number_views_fuse', type=int, default=2, help='min agree views number')
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--name', type=str, default='multi')
parser.add_argument('--run_depth', dest='run_depth', action='store_true') 
parser.add_argument('--run_fusion', dest='run_fusion', action='store_true')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


# run MVS model to save depth maps and confidence maps
def save_depth(eval_dataset):
 
    test_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False)

    # model
    model = DispMVS(args)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        for i, (imgs,Ks,Rs,Ts,depth_min,depth_max,depth_filename) in enumerate(test_loader):

            N = imgs.shape[1]
            if N<=1:
                continue
            
            device = 'cuda'
            imgs = imgs.to(device)
            Ks = Ks.to(device)
            Rs = Rs.to(device)
            Ts = Ts.to(device)
            depth_min = 1.0/depth_min.float().to(device)
            depth_max = 1.0/depth_max.float().to(device)
            depth_filename = depth_filename[0]

            # print(imgs.shape)
            b,_,_,_,h,w = imgs.shape
            init_depth = torch.rand((b,1,h//16,w//16),device=device)*(depth_min-depth_max) + depth_max
            init_depth = 1.0/init_depth

            torch.cuda.reset_max_memory_allocated()
            begin_time = time.time()
            depth_fusion = model(imgs,Ks,Rs,Ts,args.iter,init_depth,depth_min,depth_max)
            end_time = time.time()
            print('time:',end_time-begin_time)
            print('Memory Allocation:',torch.cuda.memory_allocated()/1024/1024,torch.cuda.max_memory_allocated()/1024/1024)

            depth_fusion = depth_fusion[-1]
            depth_fusion = depth_fusion.cpu().numpy()[0,0]

            depth_filename = os.path.join(args.outdir,depth_filename)

            # save depth maps
            os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
            save_pfm(depth_filename+'.pfm', depth_fusion)
            print('process:',depth_filename)
            


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) != 0:
                data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected


def check_geometric_consistency(
        depth_ref, intrinsics_ref, extrinsics_ref, 
        depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected = reproject_with_depth(
                                                        depth_ref, intrinsics_ref, extrinsics_ref,
                                                        depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < args.abs_pixel_diff, relative_depth_diff < args.relative_depth_diff)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected


def filter_depth(eval_dataset, scan_folder, out_folder, plyfilename):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:

        img_path = os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view))
        proj_path = os.path.join(scan_folder, 'cams_1/{:0>8}_cam.txt'.format(ref_view))
        depth_path = os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view))

        # skip empty depth
        if not os.path.exists(depth_path):
            continue

        # load the camera parameters
        ref_intrinsics, ref_extrinsics_r,ref_extrinsics_t,depth_min,depth_max  = eval_dataset.read_cam_file(proj_path)
        ref_extrinsics = np.zeros((4,4))
        ref_extrinsics[0:3,0:3]=ref_extrinsics_r
        ref_extrinsics[0:3,3:4]=ref_extrinsics_t
        ref_extrinsics[3,3] = 1
        # load the reference image
        ref_img = eval_dataset.read_img(img_path)
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(depth_path)[0]

        all_srcview_depth_ests = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:

            img_path = os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(src_view))
            proj_path = os.path.join(scan_folder, 'cams_1/{:0>8}_cam.txt'.format(src_view))
            depth_path = os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view))

            # skip empty depth
            if not os.path.exists(depth_path):
                continue

            # camera parameters of the source view
            src_intrinsics, src_extrinsics_r,src_extrinsics_t,_,_ = eval_dataset.read_cam_file(
                proj_path)
            src_extrinsics = np.zeros((4,4))
            src_extrinsics[0:3,0:3]=src_extrinsics_r
            src_extrinsics[0:3,3:4]=src_extrinsics_t
            src_extrinsics[3,3] = 1
            # the estimated depth of the source view
            src_depth_est = read_pfm(depth_path)[0]
            geo_mask, depth_reprojected = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est, src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)


        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)

        # remove out of range
        depth_range_mask = np.logical_and(depth_est_averaged>depth_min, depth_est_averaged<depth_max)

        # at least 3 source views matched
        geo_mask = geo_mask_sum >= args.number_views_fuse
        geo_mask = np.logical_and(geo_mask,depth_range_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)

        print("processing {}, ref-view{:0>2}, final-mask:{}".format(scan_folder, ref_view,geo_mask.mean()))

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = geo_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        color = ref_img[:, :, :][valid_points]  # hardcoded for DTU dataset
        color = color*0.5 + 0.5
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


# decide the dataset
if args.dataset == 'dtu':
    # dataset, dataloader
    eval_dataset = DTU_MVSDatasetEval(
        args.outdir, 
        args.testlist,
        'test',
        args.number_views_pred)
elif args.dataset == 'tankstemple':
    # dataset, dataloader
    eval_dataset = TanksTemple_MVSDatasetEval(
        args.outdir, 
        args.testlist,
        'test',
        args.number_views_pred)
elif args.dataset == 'eth3d':
    # dataset, dataloader
    eval_dataset = Eth3D_MVSDatasetEval(
        args.outdir, 
        args.testlist,
        'test',
        args.number_views_pred)
else:
    print("Error Wrong Dataset")

print('run_depth:',args.run_depth)
print('run_fusion:',args.run_fusion)

# generate depth map
if args.run_depth:
    save_depth(eval_dataset)

# fusion point cloud
if args.run_fusion:
    with open(args.testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]
    for scan in scans:

        if args.dataset == 'dtu':
            args.number_views_fuse   = dtu_hyper[scan][0]
            args.abs_pixel_diff      = dtu_hyper[scan][1]
            args.relative_depth_diff = dtu_hyper[scan][2]
            ply_out_path = os.path.join(args.outdir, args.name+'_{}_l3.ply'.format(scan))

        if args.dataset == 'tankstemple':
            args.number_views_fuse   = tankstemple_hyper[scan][0]
            args.abs_pixel_diff      = tankstemple_hyper[scan][1]
            args.relative_depth_diff = tankstemple_hyper[scan][2]
            ply_out_path = os.path.join(args.outdir,'{}.ply'.format(scan))
        
        print('scan info:', scan)
        print('scan info number_views_fuse:',args.number_views_fuse)
        print('scan info abs_pixel_diff:',args.abs_pixel_diff)
        print('scan info relative_depth_diff:',args.relative_depth_diff)

        # scan_id = int(scan[4:])
        scan_folder = os.path.join(args.outdir, scan)
        out_folder = os.path.join(args.outdir, scan)
        # step2. filter saved depth maps with photometric confidence maps and geometric constraints
        filter_depth(eval_dataset, scan_folder, out_folder, ply_out_path )