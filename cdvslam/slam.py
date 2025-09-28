import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from warnings import warn
from torch_scatter import scatter_mean

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net_dpv import VONet
from .net_cdv import CDVNet
from .patchgraph import PatchGraph
from .utils import *
# from .ba import BA as PYBA

mp.set_start_method('spawn', True)


autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class SLAM:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network)
        self.get_seg_head(cfg.SEG_HEAD)
        self.get_depth_head(cfg.DEPTH_HEAD)
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        DIMF = self.DIMF
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = mp.Value('i', 0) # number of ALL processed frames

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, DIMF, self.P, self.P, **kwargs)
        do_seg = (self.seg_head is not None)
        do_scale = (self.depth_head is not None)
        self.last_scale_adjustment = 0
        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, seg=do_seg, scale=do_scale, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, DIMF, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, DIMF, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        self.terminating = False
        if viz:
            self.start_viewer()

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = CDVNet(dino_adapt=False)
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network
        self.network.update.M = self.cfg.PATCHES_PER_FRAME

        # steal network attributes
        self.DIM = self.network.DIM
        if hasattr(self.network, 'DIMF'):
            self.DIMF = self.network.DIMF
        else:
            self.DIMF = 128
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def get_seg_head(self, seg_head):
        self.seg_head = None
        if isinstance(self.network, VONet):
            return
        if not seg_head or seg_head == 'None':
            return
        if isinstance(seg_head, str):
            from .net_cdv import SegHeadFlatten
            if 'ade20k' in seg_head:
                self.num_classes = 150
                from DINO_modules.datamaps import ADE20K_INDEX_FILT
                self.filt_index = torch.tensor(ADE20K_INDEX_FILT, dtype=torch.uint8, device='cuda') - 1
            elif 'voc2012' in seg_head:
                self.num_classes = 21
                self.filt_index = torch.tensor([0], dtype=torch.uint8, device='cuda')
            else:
                self.num_classes = None
            self.seg_head = SegHeadFlatten(num_classes=self.num_classes)
            self.seg_head.load_state_dict(torch.load(
                seg_head
            ))

        elif isinstance(seg_head, torch.nn.Module):
            self.seg_head = seg_head
        print("Loaded seg head:", seg_head)

        self.seg_head.filt_index = self.filt_index
        self.seg_head.cuda()
        self.seg_head.eval()

    def get_depth_head(self, depth_head):
        self.depth_head = None
        if isinstance(self.network, VONet):
            return
        if not depth_head or depth_head == 'None':
            return
        if 'linear' in depth_head:
            from .net_cdv import DepthHeadFlatten
            self.depth_head = DepthHeadFlatten()
            self.depth_head.load_state_dict(torch.load(depth_head))
        elif 'dpt' in depth_head:
            from DINO_modules.hub.dpt.depth_head_mm import DPTHead
            ckpt = torch.load('dinov2_vits14_nyu_dpt_head.pth')
            key_map = lambda k : k.replace('decode_head.', '') if 'decode_head.' in k else k
            state_newkey = {
                    key_map(k) : v for k, v in ckpt['state_dict'].items()
            }
            self.depth_head = DPTHead()
            self.depth_head.load_state_dict(state_newkey)
        else:
            raise NotImplementedError
        print("Loaded depth head:", depth_head)
        self.scale_factor = self.depth_head.scale_factor = self.depth_head.max_depth / 4. #2.
        self.depth_head.cuda()
        self.depth_head.eval()

    def _win_mp_fix(self, start=True):
        """Work around for Windows torch multiprocessing bug: https://github.com/pytorch/pytorch/issues/156618"""
        import sys
        if sys.platform.startswith('win'):
            print('Runing on Windows')
            if start:
                self.network.cpu()
                if self.seg_head is not None:
                    self.seg_head.cpu()
                if self.depth_head is not None:
                    self.depth_head.cpu()
            else :
                self.network.cuda()
                if self.seg_head is not None:
                    self.seg_head.cuda()
                if self.depth_head is not None:
                    self.depth_head.cuda()
                self.pg.poses_[:,6] = 1.0
        else:
            return

    def start_viewer(self):
        from .o3dviewer import O3DViewer

        self.tcounts = torch.from_numpy(self.pg.tstamps_).cuda().share_memory_()

        self._win_mp_fix(start=True)
        self.viewer = O3DViewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_.view(self.N, self.M, 3),
            self.pg.colors_,
            self.pg.intrinsics_,
            self.pg.dirty_,
            self.pg.weight_,
            self.pg.seg_,
            self.tcounts,
            self.pg.n,)
        self._win_mp_fix(start=False)

        self.pg.dirty_[0] = True

        self.pg.viewer = self.viewer
        
    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, self.DIMF, 3, 3)

    @property
    def n(self):
        return self.pg.n.value

    @n.setter
    def n(self, val):
        with self.pg.n.get_lock():
            self.pg.n.value = val

    @property
    def m(self):
        return self.pg.m.value

    @m.setter
    def m(self, val):
        with self.pg.m.get_lock():
            self.pg.m.value = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        pose = dP * self.get_pose(t0)
        if self.terminating:
            self.traj[t] = pose.data
        return pose

    def terminate(self):
        if not self.is_initialized:
            warn('SLAM terminates without initialization! Frame count: ' + str(self.counter.value))
            poses = SE3.Identity(self.counter.value).data.numpy()
            tstamps = np.array(self.tlist, dtype=np.float64)
            return poses, tstamps

        self.terminating = True

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter.value)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        self.terminating = False
        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def get_encoding(self, kk):
        if not hasattr(self.network, 'posenc'):
            warn('No position encoding')
            return None
        # warn('Computing position encoding')
        p = self.patches[:, kk, :2, 3//2, 3//2] 
        h = self.ht // self.RES
        w = self.wd // self.RES
        p_shift = torch.tensor([w/2, h/2]).to(p).view(1,1,2)
        p_scale = max(w/2, h/2)
        p_n = (p - p_shift) / p_scale
        posenc_cache = self.network.posenc(p_n)
        return posenc_cache

    def preprocess_image(self, image):
        if not hasattr(self.network, 'simple_preprocess') or \
            self.network.simple_preprocess == False:
            # warn('Using DPVO image preprocessing')
            image = 2 * (image[None,None] / 255.0) - 0.5
        else:
            # warn('Using simple image preprocessing')
            image = image[None,None] / 255.0
        return image

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        posenc = self.get_encoding(kk)

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk, posenc_cache=posenc)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        with Timer('motionmag', enabled=self.enable_timing):
            i = self.n - self.cfg.KEYFRAME_INDEX - 1
            j = self.n - self.cfg.KEYFRAME_INDEX + 1
            m = self.motionmag(i, j) + self.motionmag(j, i)

        with Timer('key_rm', enabled=self.enable_timing):
            if m / 2 < self.cfg.KEYFRAME_THRESH:
                k = self.n - self.cfg.KEYFRAME_INDEX
                t0 = self.pg.tstamps_[k-1]
                t1 = self.pg.tstamps_[k]

                dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
                # self.pg.delta[t1] = (t0, dP)
                self.pg.add_delta(t1, t0, dP, k-1)

                to_remove = (self.pg.ii == k) | (self.pg.jj == k)
                self.remove_factors(to_remove, store=False)

                self.pg.kk[self.pg.ii > k] -= self.M
                self.pg.ii[self.pg.ii > k] -= 1
                self.pg.jj[self.pg.jj > k] -= 1

                for i in range(k, self.n-1):
                    self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                    self.pg.colors_[i] = self.pg.colors_[i+1]
                    self.pg.poses_[i] = self.pg.poses_[i+1]
                    self.pg.patches_[i] = self.pg.patches_[i+1]
                    self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                    self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                    self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                    self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                    self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]
                if self.viewer is not None:
                    with self.pg.n.get_lock():
                        self.pg.dirty_[k: self.n-1] = True
                        self.viewer.add_delta(t1, t0, dP, k-1)

                self.n -= 1
                self.m-= self.M

                if self.cfg.CLASSIC_LOOP_CLOSURE:
                    self.long_term_lc.keyframe(k)

            to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
            if self.cfg.LOOP_CLOSURE:
                # ...unless they are being used for loop closure
                lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
                to_remove = to_remove & ~lc_edges
            self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)

        self.ran_global_ba[self.n] = True
        if self.viewer is not None:
            with self.pg.n.get_lock():
                self.pg.dirty_[t0:self.n] = True

    def update(self):
        with Timer("flow", enabled=self.enable_timing):
            coords = self.reproject()

            posenc = self.get_encoding(self.pg.kk)
            
            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk, posenc_cache=posenc)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float() # b, n_patches, 2
            target = coords[...,self.P//2,self.P//2] + delta.float()

            filtered_weight, target = self.update_weight(weight, target)
        
        if self.viewer is not None:
            if self.cfg.VIEW_FILTERED_WEIGHT:
                view_weight = filtered_weight[0].mean(dim=-1)
            else:
                view_weight = weight[0].mean(dim=-1)
            self.update_viewer_weight(view_weight)

        with Timer("BA", enabled=self.enable_timing):
            # run global bundle adjustment if there exist long-range edges
            if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                if self.viewer is not None:
                    print('global BA')
                self.__run_global_BA()
            else:
                t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                t0 = max(t0, 1)
                fastba.BA(self.poses, self.patches, self.intrinsics, 
                    target, filtered_weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)

                if self.depth_head:
                    self.scale_adjustment(t0, w=(1-weight.mean()))

                if self.viewer is not None:
                    with self.pg.n.get_lock():
                        self.pg.dirty_[t0:self.n] = True

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m]) # (1,?,3,3,4)
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def semantic_seg(self, imap):
        if self.seg_head:
            imap = imap.view(self.M, self.DIM) * 4.
            seg_logit = self.seg_head(imap)
            seg_class = torch.argmax(seg_logit, dim=-1, keepdim=False)
            seg_class = seg_class.to(torch.uint8)
            with self.pg.n.get_lock():
                self.pg.seg_[self.n] = seg_class

            return seg_class
        
    def mono_depth(self, imap, token_cls):
        if self.depth_head:
            imap = imap.view(self.M, self.DIM) * 4
            assert len(token_cls.shape) == 2 and token_cls.shape[0] == 1
            token_cls = token_cls.expand_as(imap)
            tokens = torch.cat([imap, token_cls], dim=-1)
            depth = self.depth_head(tokens) # (n, 1)
            return depth
        else:
            return None

    def scale_adjustment(self, t0, w=0.2):
        P1 = SE3(self.pg.poses_[self.n-2])
        P2 = SE3(self.pg.poses_[self.n-1])
        angle = (P1 * P2.inv()).log()[3:].norm().item()
        if angle < 0.04:
            return 

        disps = self.pg.patches_[t0:self.n, :, 2, self.P//2, self.P//2]

        s = torch.mean(disps)
        rate = torch.mean(self.pg.scale_[t0:self.n]) / s

        rate = 1. + w  * (rate - 1) 
        self.pg.patches_[t0:self.n, :, 2] *= rate
        Ps = SE3(self.pg.poses_[t0-1 : self.n]).inv()
        xyz = Ps.data[:, :3]
        xyz[1:] += (1 - (1/rate)) * (xyz[0:1] - xyz[1:])
        self.pg.poses_[t0:self.n, :3] = Ps[1:].inv().data[:, :3]
        self.last_scale_adjustment = self.n

    def update_weight(self, weight, target):
        if self.seg_head and self.cfg.FILTER_DYNAMIC_CLASS:
            # warn('Using seg head to filter dynamic classes')
            filt_mask = torch.isin(self.pg.seg_[:self.n], self.filt_index)
            multiplyer = torch.where(filt_mask, 1e-4, 1.)
            weight = weight * multiplyer.view(1, self.n * self.M, 1)[:, self.pg.kk]

        self.pg.target = target
        self.pg.weight = weight
        return weight, target

    def update_viewer_weight(self, view_weight):
        try:
            with self.pg.n.get_lock():
                kk_min, kk_max = self.pg.kk.min(), self.pg.kk.max()
                kk_0 = self.pg.kk - kk_min
                kk_unique = torch.unique(self.pg.kk)
                kk_0_unique = kk_unique - kk_min

                self.pg.weight_.view(-1)[kk_unique] =  \
                    scatter_mean(view_weight, kk_0, dim=0).half()[kk_0_unique]
        except RuntimeError as e:
            kk_uni, sw = self.pg.kk.unique(), scatter_mean(view_weight[0].mean(dim=-1), self.pg.kk, dim=0)
            print(view_weight.shape, self.pg.weight_.shape)
            print(print(kk_uni.shape, sw.shape, kk_uni.max()))
            print(self.pg.kk.max() - self.pg.kk.min() + 1)
            raise e
    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        if self.viewer is not None:
            image_raw = image
        assert len(image.shape) == 3
        
        with Timer('patch', enabled=self.enable_timing):
            image = self.preprocess_image(image)
            with autocast(enabled=self.cfg.MIXED_PRECISION):
                fmap, gmap, imap, patches, _, clr, data_dict = \
                    self.network.patchify(image,
                        patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                        centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                        centroid_sample_strat=self.cfg.CENTROID_SAMPLE_STRAT,
                        return_color=True,
                        seg_head=self.seg_head,
                        depth_head=self.depth_head)

                self.semantic_seg(imap)

        with Timer('state', enabled=self.enable_timing):
            ### update state attributes ###
            self.tlist.append(tstamp)
            self.pg.tstamps_[self.n] = self.counter.value
            self.pg.intrinsics_[self.n] = intrinsics / self.RES

            # color info for visualization
            clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
            self.pg.colors_[self.n] = clr.to(torch.uint8)

            self.pg.index_[self.n + 1] = self.n + 1
            self.pg.index_map_[self.n + 1] = self.m + self.M

            if self.n > 1:
                if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                    P1 = SE3(self.pg.poses_[self.n-1])
                    P2 = SE3(self.pg.poses_[self.n-2])

                    # To deal with varying camera hz
                    *_, a,b,c = [1]*3 + self.tlist
                    fac = (c-b) / (b-a)

                    xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                    tvec_qvec = (SE3.exp(xi) * P1).data
                    self.pg.poses_[self.n] = tvec_qvec
                else:
                    tvec_qvec = self.poses[self.n-1]
                    self.pg.poses_[self.n] = tvec_qvec

            if self.depth_head:
                self.pg.scale_[self.n] = torch.mean(patches[0,:,2])
            else:
                patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
                if self.is_initialized:
                    s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
                    patches[:,:,2] = s

            self.pg.patches_[self.n] = patches

            ### update network attributes ###
            self.imap_[self.n % self.pmem] = imap.squeeze()
            self.gmap_[self.n % self.pmem] = gmap.squeeze()
            self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
            self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

            with self.counter.get_lock():
                self.counter.value += 1        
            if self.n > 0 and not self.is_initialized:
                if self.motion_probe() < self.cfg.MOTION_PROBE_THR: # default: 2.0
                    # self.pg.delta[self.counter.value - 1] = (self.counter.value - 2, Id[0])
                    self.pg.add_delta(self.counter.value - 1, self.counter.value - 2, Id[0])
                    if self.viewer is not None:
                        self.tcounts[:self.n] = torch.from_numpy(self.pg.tstamps_[:self.n]).cuda()
                        self.viewer.update_image(image_raw.contiguous())
                        self.viewer.add_delta(self.counter.value - 1, self.pg.tstamps_[self.n-1], Id[0], self.n-1)
                    return 'Not keyframe'

            self.n += 1
            self.m += self.M

            if self.cfg.LOOP_CLOSURE:
                if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                    """ Add loop closure factors """
                    lii, ljj = self.pg.edges_loop()
                    if lii.numel() > 0:
                        self.last_global_ba = self.n
                        self.append_factors(lii, ljj)

            # Add forward and backward factors
            self.append_factors(*self.__edges_forw())
            self.append_factors(*self.__edges_back())

        with Timer('update', enabled=self.enable_timing):
            if self.n == 8 and not self.is_initialized:
                self.is_initialized = True

                for itr in range(12):
                    self.update()

            elif self.is_initialized:
                self.update()
                self.keyframe()

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                try:
                    lc_count0 = self.long_term_lc.lc_count
                    self.long_term_lc.attempt_loop_closure(self.n)
                    self.long_term_lc.lc_callback()
                    if self.viewer is not None and self.long_term_lc.lc_count > lc_count0:
                        print('Performed classical lc')
                except IndexError as e:
                    print(e)

        if self.viewer is not None:
            self.tcounts[:self.n] = torch.from_numpy(self.pg.tstamps_[:self.n]).cuda()
            self.viewer.update_image(image_raw.contiguous(), 
                                     self.pg.patches_[self.n-1], self.pg.weight_[self.n-1],
                                     self.pg.seg_[self.n-1] if self.seg_head else None,
                                     self.RES,
                                     save_dir=None,
                                    )                       
        return 'Is keyframe'