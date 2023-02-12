import cv2
import numpy as np

def viz_depth(path_name,depth,mask,depth_min=None,depth_max=None):

    depth_viz = depth[0,0,:,:].detach().cpu().numpy()
    mask_viz = mask[0,0,:,:].detach().cpu().numpy()

    if depth_min is None or depth_max is None:
        depth_min_max = depth_viz[mask_viz>0]
        depth_min = depth_min_max.min()
        depth_max = depth_min_max.max()

    depth_rng = (depth_max-depth_min)
    depth_viz = 255*(depth_viz-depth_min)/depth_rng
    depth_viz = depth_viz*mask_viz
    depth_viz = depth_viz.astype(np.uint8)
    cv2.imwrite(path_name,cv2.applyColorMap(depth_viz,cv2.COLORMAP_JET))

    return depth_min,depth_max

def viz_diff(path_name,depth,mask):

    depth_viz = depth[0,0,:,:].detach().cpu().numpy()
    depth = depth.clip(0,255)
    depth_viz = depth_viz*mask[0,0].detach().cpu().numpy()
    depth_viz = depth_viz.astype(np.uint8)
    cv2.imwrite(path_name,cv2.applyColorMap(depth_viz,cv2.COLORMAP_JET))
