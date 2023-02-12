import torch
import torch.nn as nn
import torch.nn.functional as F

from update import SmallUpdateBlock,BasicUpdateBlock
from extractor import SmallEncoder,BasicEncoder
from corr import GuidedCorrBlock

class DispMVS(nn.Module):

    def __init__(self,args):
        super(DispMVS, self  ).__init__()

        self.args = args

        # feature network, context network, and update block
        self.hdim = [96,128]
        self.cdim = [96,128]
        self.corr_levels = [2,4]
        self.corr_radius = [2,4]

        self.fnet = BasicEncoder(output_dim=[96,128], deform=False, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=[96*2,128*2], deform=True, norm_fn='batch', dropout=args.dropout)
        self.update_block_4 = BasicUpdateBlock(hidden_dim=self.hdim[1], corr_level=self.corr_levels[1], corr_radiu=self.corr_radius[1])
        self.update_block_2 = BasicUpdateBlock(hidden_dim=self.hdim[0], corr_level=self.corr_levels[0], corr_radiu=self.corr_radius[0])
        
        self.xyz = None
        self.depth_min = None
        self.depth_max = None

    def upsample_flow(self, flow, mask, up_scale):

        """ Upsample flow field [H/4, W/4, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, up_scale, up_scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3,3], padding=1)
        del flow
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        del mask
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 1, up_scale*H, up_scale*W)

    # basic
    def gen_grid(self, batch, height, width, device):
        
        return torch.cat([
            torch.arange(0, width).view(1, 1, width, 1, 1).expand(1, height, width, 1, 1).expand(batch, height, width, 1, 1),
            torch.arange(0, height).view(1, height, 1, 1, 1).expand(1, height, width, 1, 1).expand(batch, height, width, 1, 1),
            torch.ones(batch,height,width,1, 1)],
            dim=3).to(device).detach()

    # dealing with K and R, T
    def scale_K(self, Ks, scale):
        
        Ks_scale = []

        for i in range(self.n):
       
            K_scale = Ks[:,i,:,:,:,:]/scale
            K_scale[:,:,:,2,2] = 1.0
            Ks_scale.append(K_scale)
        
        return Ks_scale

    def relative_pose(self, R, T):

        # shape
        # [B,N,1,1,3,3]

        R_rela = []
        T_rela = []

        # ref
        R_ref = R[:,0,:,:,:,:]
        T_ref = T[:,0,:,:,:,:]

        R_rela.append(None)
        T_rela.append(None)

        for i in range(1,self.n):

            # nei
            R_nei = R[:,i,:,:,:,:]
            T_nei = T[:,i,:,:,:,:]

            # relative
            r = R_nei@R_ref.permute(0,1,2,4,3)
            t = T_nei - r@T_ref

            R_rela.append(r)
            T_rela.append(t)

        return R_rela,T_rela

    # flow and depth
    def proj_depth(self,K_nei,R_nei,T_nei,depth):

        # B C H W -> B H W C 1
        # depth = depth.permute(0,2,3,1).unsqueeze(4)

        # shap [B,H,W,3,1]
        xyz = self.xyz * depth.permute(0,2,3,1).unsqueeze(4)
        del depth
        xyz = R_nei@xyz + T_nei
        xyz = K_nei@xyz

        # shape [B,H,W,2]
        proj = xyz[:,:,:,0:2,0]/( (xyz[:,:,:,2:3,0]).abs() + 1e-6 )
        del xyz
        proj = proj.permute(0,3,1,2)

        return proj

    def calc_flow(self,K_nei,R_nei,T_nei,depth):
        
        proj_sta = self.proj_depth(K_nei,R_nei,T_nei,depth)
        proj_end = self.proj_depth(K_nei,R_nei,T_nei,depth*2+10) # add ten to avoid depth_min = depth_min*2
        del depth

        flow_dir = proj_end - proj_sta
        del proj_end

        flow_dir = flow_dir/flow_dir.norm(dim=1,keepdim=True)

        return proj_sta,flow_dir

    def flow2dep(self,K_nei,R_nei,T_nei,match,flow_mask):

        # ref
        r_xyz = R_nei@self.xyz
        rx = r_xyz[:,:,:,0:1,:]
        ry = r_xyz[:,:,:,1:2,:]
        rz = r_xyz[:,:,:,2:3,:]
        del r_xyz

        # nei
        b,_,h,w = match.shape
        device = match.device

        n_xyz = torch.cat([
            match.permute(0,2,3,1).unsqueeze(4),
            torch.ones((b,h,w,1,1),device=device)],dim=3)
        del match

        n_xyz = K_nei.inverse()@n_xyz
        n_xyz = n_xyz/( (n_xyz[:,:,:,2:3,:]).abs() + 1e-6 )
        nx = n_xyz[:,:,:,0:1,:]
        ny = n_xyz[:,:,:,1:2,:]
        # nz = n_xyz[:,:,:,2:3,:]
        del n_xyz

        # depth
        tx = T_nei[:,:,:,0:1,:]
        ty = T_nei[:,:,:,1:2,:]
        tz = T_nei[:,:,:,2:3,:]

        # as d always larger than zero
        # use abs and 1e-6 to avoid divide zero
        # dx = ( tz*nx - tx ).abs() / ( ( rx -  rz*nx ).abs() + 1e-6 )
        # dy = ( tz*ny - ty ).abs() / ( ( ry -  rz*ny ).abs() + 1e-6 )
        # dx = dx.squeeze(4).permute(0,3,1,2)
        # dy = dy.squeeze(4).permute(0,3,1,2) 
        # d = torch.cat([dx,dy],dim=1)

        # inv_dx = ( ( rx -  rz*nx ).abs()  ) / ( ( tz*nx - tx ).abs() + 1e-6 )
        # inv_dy = ( ( ry -  rz*ny ).abs()  ) / ( ( tz*ny - ty ).abs() + 1e-6 )
        # inv_dx = inv_dx.squeeze(4).permute(0,3,1,2)
        # inv_dy = inv_dy.squeeze(4).permute(0,3,1,2) 
        inv_d = torch.cat([
            ( ( rx -  rz*nx ).abs()  / ( ( tz*nx - tx ).abs() + 1e-6 ) ).squeeze(4).permute(0,3,1,2),
            ( ( ry -  rz*ny ).abs()  / ( ( tz*ny - ty ).abs() + 1e-6 ) ).squeeze(4).permute(0,3,1,2),
            ],dim=1)

        del rx,ry,rz
        del tx,ty,tz 
        del nx,ny

        # select bigger one
        inv_d = torch.gather(inv_d,dim=1,index=flow_mask)
        del flow_mask

        # norm
        inv_d = (inv_d-self.depth_max)/(self.depth_min-self.depth_max)
        inv_d = inv_d.clip(0,1)

        return inv_d
    
    def depth_estimate(self,index,scale,up_scale,update_block,feat,net,inp,Ks,Rs,Ts,iter,init_depth):

        # relative
        Ks_scale = self.scale_K(Ks,scale)
        Rs_rela, Ts_rela = self.relative_pose(Rs, Ts)

        b,_,h,w = net.shape
        device  = net.device

        # xyz
        self.xyz = Ks_scale[0].inverse()@self.gen_grid(b,h,w,device)
        self.xyz = self.xyz/self.xyz[:,:,:,2:3,:]

        # init matching function
        net_s = [None]
        flow_s = [None]
        corr_fn_s = [None]
        ref_feat = feat[0]
        for nei_id in range(1,self.n):
           
            flow_bas,flow_dir = self.calc_flow(Ks_scale[nei_id],Rs_rela[nei_id],Ts_rela[nei_id],init_depth)
            corr_fn_s.append(GuidedCorrBlock(feat[nei_id],flow_bas,flow_dir,self.corr_levels[index],self.corr_radius[index]) )
            del flow_bas,flow_dir

            flow_s.append(torch.zeros((b,1,h,w),device=device).detach())

            net_s.append(net)

        del net,feat,init_depth

        # depth estimation
        if self.training:
            depth_iter = []
        else:
            depth_final = []

        for t in range(iter):
            
            depth_fusion = []
            conf_fusion = []

            for nei_id in range(1,self.n):
                
                # remove grad
                flow_s[nei_id] = flow_s[nei_id].detach()

                net_s[nei_id], delta_flow, delta_mask = update_block(
                        net_s[nei_id], 
                        inp, 
                        corr_fn_s[nei_id].get_cost(ref_feat,flow_s[nei_id]), 
                        flow_s[nei_id])

                # update flow
                flow_s[nei_id] = flow_s[nei_id] + delta_flow[:,0:1,:,:]

                # convert flow to depth
                inv_depth = self.flow2dep(
                        Ks_scale[nei_id],
                        Rs_rela[nei_id],Ts_rela[nei_id],
                        corr_fn_s[nei_id].get_match(flow_s[nei_id]),
                        corr_fn_s[nei_id].flow_mask)

                inv_depth = self.upsample_flow(inv_depth,delta_mask,up_scale)
                conf = self.upsample_flow(delta_flow[:,1:2,:,:],delta_mask,up_scale)
                del delta_flow,delta_mask

                depth_fusion.append(inv_depth)
                conf_fusion.append(conf)

                del conf,inv_depth

            # inverse fusion
            depth_fusion = torch.cat(depth_fusion,dim=1)
            conf_fusion = torch.cat(conf_fusion,dim=1)
            conf_fusion = F.softmax(conf_fusion,dim=1)
            depth_fusion = (depth_fusion*conf_fusion).sum(dim=1,keepdim=True)

            del conf_fusion

            # re-norm
            depth_fusion = depth_fusion*(self.depth_min-self.depth_max) + self.depth_max
            depth_fusion = 1.0/depth_fusion

            # output
            if self.training:
                depth_iter.append(depth_fusion)
            else:
                if t==(iter-1):
                    depth_final.append(depth_fusion)

            # dep2flow
            if t<(iter-1):
                depth_fusion = F.interpolate(depth_fusion,scale_factor=1.0/up_scale,recompute_scale_factor=True)
                for nei_id in range(1,self.n):
                    flow_s[nei_id] = corr_fn_s[nei_id].get_flow(
                            self.proj_depth(Ks_scale[nei_id],Rs_rela[nei_id],Ts_rela[nei_id],depth_fusion))

        del depth_fusion
        del self.xyz,ref_feat
        del net_s,flow_s,corr_fn_s
        del Ks_scale,Rs_rela,Ts_rela

        if self.training:
            return depth_iter
        else:
            return depth_final

    def forward(self,imgs,Ks,Rs,Ts,iter,init_depth,depth_min,depth_max):

        hdim = self.hdim
        cdim = self.cdim

        self.n = imgs.shape[1]
        self.depth_min = depth_min
        self.depth_max = depth_max

        # feature network
        feat_2 = []
        feat_4 = []
        for i in range(self.n):
            f2,f4 = self.fnet(imgs[:,i,0,:,:,:])

            feat_2.append(f2)
            feat_4.append(f4)

            del f2,f4 
        
        # context network
        ceat_2,ceat_4 = self.cnet(imgs[:,0,0,:,:,:])
        del imgs

        net_2, inp_2 = torch.split(ceat_2, [hdim[0], cdim[0]], dim=1)
        net_2 = torch.tanh(net_2)
        inp_2 = torch.relu(inp_2)
        del ceat_2

        net_4, inp_4 = torch.split(ceat_4, [hdim[1], cdim[1]], dim=1)
        net_4 = torch.tanh(net_4)
        inp_4 = torch.relu(inp_4)
        del ceat_4

        # B,C,H/16,W/16-->B,C,H/4,W/4
        depth_2 = self.depth_estimate(1,2**4,4,self.update_block_4,feat_4,net_4,inp_4,Ks,Rs,Ts,iter[1],init_depth)
        del feat_4,net_4,inp_4
        del init_depth

        # B,C,H/4,W/4-->B,C,H,W
        depth_0 = self.depth_estimate(0,2**2,4,self.update_block_2,feat_2,net_2,inp_2,Ks,Rs,Ts,iter[0],depth_2[-1].detach())
        del feat_2,net_2,inp_2

        if self.training:
            return depth_0,depth_2
        else:
            del depth_2
            return depth_0