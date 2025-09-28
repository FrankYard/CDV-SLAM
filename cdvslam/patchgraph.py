import numpy as np
import torch
from einops import asnumpy, reduce, repeat
from torch.multiprocessing import Value

from . import projective_ops as pops
from .lietorch import SE3
from .loop_closure.optim_utils import reduce_edges
from .utils import *


class PatchGraph:
    """ Dataclass for storing variables """

    def __init__(self, cfg, P, DIM, pmem, seg=False, scale=False, **kwargs):
        self.cfg = cfg
        self.P = P
        self.pmem = pmem
        self.DIM = DIM

        self.n = Value('i', 0)      # number of frames
        self.m = Value('i', 0)      # number of patches

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.tstamps_ = np.zeros(self.N, dtype=np.int64)
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda").share_memory_()
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda").share_memory_()

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda").share_memory_()
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda").share_memory_()
        if seg:
            self.seg_ = torch.zeros(self.N, self.M, dtype=torch.uint8, device='cuda').share_memory_()
        else:
            self.seg_ = None

        if scale:
            self.scale_ = torch.zeros(self.N, dtype=torch.float, device='cuda')
            self.scale_factor = 1.
        else:
            self.scale_ = None
            self.scale_factor = None

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        self.dirty_ = torch.zeros(self.N, dtype=torch.bool, device='cuda').share_memory_()
        self.weight_ = torch.ones(self.N, self.M, dtype=torch.float16, device='cuda').share_memory_()

        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        ### edge information ###
        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")

        ### inactive edge information (i.e., no longer updated, but useful for BA) ###
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.weight_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")
        self.target_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")

    def edges_loop(self):
        """ Adding edges from old patches to new frames """
        lc_range = self.cfg.MAX_EDGE_AGE
        l = self.n.value - self.cfg.REMOVAL_WINDOW # l is the upper bound for "old" patches

        if l <= 0:
            return torch.empty(2, 0, dtype=torch.long, device='cuda')

        # create candidate edges
        jj, kk = flatmeshgrid(
            torch.arange(self.n.value - self.cfg.GLOBAL_OPT_FREQ, self.n.value - self.cfg.KEYFRAME_INDEX, device="cuda"),
            torch.arange(max(l - lc_range, 0) * self.M, l * self.M, device="cuda"), indexing='ij')
        ii = self.ix[kk]

        # Remove edges which have too large flow magnitude
        flow_mg, val = pops.flow_mag(SE3(self.poses), self.patches[...,1,1].view(1,-1,3,1,1), self.intrinsics, ii, jj, kk, beta=0.5)
        flow_mg_sum = reduce(flow_mg * val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).float()
        num_val = reduce(val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).clamp(min=1)
        flow_mag = torch.where(num_val > (self.M * 0.75), flow_mg_sum / num_val, torch.inf)

        mask = (flow_mag < self.cfg.BACKEND_THRESH)
        es = reduce_edges(asnumpy(flow_mag[mask]), asnumpy(ii[::self.M][mask]), asnumpy(jj[::self.M][mask]), max_num_edges=1000, nms=1)

        edges = torch.as_tensor(es, device=ii.device)
        ii, jj = repeat(edges, 'E ij -> ij E M', M=self.M, ij=2)
        kk = ii.mul(self.M) + torch.arange(self.M, device=ii.device)
        return kk.flatten(), jj.flatten()

    def normalize(self):
        """ normalize depth and poses """
        s = self.patches_[:self.n.value,:,2].mean()
        if self.scale_ is not None:
            s /= self.scale_[:self.n.value].mean()
        self.patches_[:self.n.value,:,2] /= s
        self.poses_[:self.n.value,:3] *= s

        for t, (t0, dP) in self.delta.items():
            # self.delta[t] = (t0, dP.scale(s))
            self.add_delta(t, t0, dP.scale(s))
        if hasattr(self, 'viewer'):
            for t, (t0, dP) in self.delta.items():
                self.viewer.add_delta(t, t0, dP)
        self.poses_[:self.n.value] = (SE3(self.poses_[:self.n.value]) * SE3(self.poses_[[0]]).inv()).data

        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m.value], self.intrinsics, self.ix[:self.m.value])
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
        self.points_[:len(points)] = points[:]
        with self.n.get_lock():
            self.dirty_[:self.n.value] = True

    def add_delta(self, t, t0, dP, kid=None):
        self.delta[t] = (t0, dP)

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)
