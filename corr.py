import torch
import torch.nn.functional as F

def sampler(img, coords):

    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]

    xgrid, ygrid = coords.split([1,1], dim=-1)
    del coords

    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    del xgrid,ygrid

    img = F.grid_sample(img, grid, align_corners=True)
    del grid

    return img

# used only for images with relative poses
class GuidedCorrBlock:

    def __init__(self, 
                fmap2, 
                flow_bas, flow_dir, 
                corr_levels, corr_radius):

        self.D = fmap2.shape[1]
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        self.fmap2_pyra = [fmap2]
        for i in range(1,self.corr_levels):
            self.fmap2_pyra.append(F.avg_pool2d(self.fmap2_pyra[-1], 2, stride=2))

        self.flow_bas = flow_bas
        self.flow_dir = flow_dir
        self.flow_mask = (flow_dir.detach().abs()).max(dim=1,keepdim=True)[1]

    def get_flow(self,match):                                                                                                                                                                                                                                                                                                                                     

        match -= self.flow_bas
        match = ( match * self.flow_dir ).sum(dim=1,keepdim=True)
       
        return match

    def get_match(self, flow):

        return self.flow_bas + flow*self.flow_dir

    def get_cost(self, fmap1_pyra, flow):

        corrs = []

        for p in range(self.corr_levels):
            
            cur_scale = 2**p

            for s in range(-self.corr_radius,self.corr_radius+1):

                match = self.flow_bas +  ( flow + s )*self.flow_dir
                match = match.permute(0, 2, 3, 1) / cur_scale
                # fmap2_warp = sampler(self.fmap2_pyra[p],match)
                corr_s = ( fmap1_pyra*sampler(self.fmap2_pyra[p],match) ).sum(dim=1,keepdims=True)
                del match

                corr_s = corr_s/self.D
                corrs.append(corr_s)
                del corr_s
        
        del flow,fmap1_pyra
        
        corrs = torch.cat(corrs,dim=1)

        return corrs





