import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

from . import fastba
from . import altcorr
from .lietorch import SE3

from .extractor import load_model
from .blocks import GradientClip, GatedResidual, SoftAgg
from .att_layers.layers import GatedAttention, LearnableFourierPositionalEncoding, EncoderLayer

from .utils import *
from .ba import BA
from . import projective_ops as pops

from DINO_modules.hub.backbones import _make_dinov2_model
from DINO_modules.hub.utils import Padding

autocast = torch.cuda.amp.autocast

DIMI = 384
DIMF = 24
SCALEI = 14. # TODO: HOW TO CHECKE IT ?
SCALEF = 4.

class Update(nn.Module):
    def __init__(self, p, version='pa'):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIMI, DIMI),
            nn.ReLU(inplace=True),
            nn.Linear(DIMI, DIMI))

        self.c2 = nn.Sequential(
            nn.Linear(DIMI, DIMI),
            nn.ReLU(inplace=True),
            nn.Linear(DIMI, DIMI))
        
        self.norm = nn.LayerNorm(DIMI, eps=1e-3)

        self.agg_kk = SoftAgg(DIMI)

        self.version = version

        if self.version == 'dpvo':
            self.agg_ij = SoftAgg(DIMI)

            self.gru = nn.Sequential(
                nn.LayerNorm(DIMI, eps=1e-3),
                GatedResidual(DIMI),
                nn.LayerNorm(DIMI, eps=1e-3),
                GatedResidual(DIMI),
            )
        elif self.version == 'a':
            self.atten = EncoderLayer(DIMI, nhead=8, attention='linear')
        elif self.version == 'pa':
            self.norm2 = nn.LayerNorm(DIMI, eps=1e-3)
            self.gatten = GatedAttention(DIMI, nhead=8, attention='linearv2')
        else:
            raise NotImplementedError

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIMI),
            nn.ReLU(inplace=True),
            nn.Linear(DIMI, DIMI),
            nn.LayerNorm(DIMI, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIMI, DIMI),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIMI, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIMI, 2),
            GradientClip(),
            nn.Sigmoid())

    @classmethod
    def check_inds(self, ii, jj, ij_ind):
        # _ii = ii.view(ni, ppi, nj)
        # _jj = jj.view(ni, ppi, nj)
        # assert (_ii[:, :1, :1] == _ii).all()
        # assert (_jj[:1, :1, :] == _jj).all()
        _ii, _jj = ii[ij_ind], jj[ij_ind]
        assert (_ii[:, :1] == _ii).all()
        assert (_jj[:, :1] == _jj).all()

    def forward(self, net, inp, corr, flow, ii, jj, kk, ij_ind=None, posenc_cache=None):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        # net = net + self.agg_ij(net, ii*12345 + jj)

        if ij_ind is None:
            ij = ii*12345 + jj
            ij_ind = torch.sort(ij)[1]
            M = (ij == ij[0]).sum().item()
            ij_ind = ij_ind.view(-1, M).sort(dim=-1)[0]
        else:
            M = ij_ind.size(1)
            ni, nj = len(torch.unique(ii)), len(torch.unique(jj))
            _ppi = len(kk) // (ni * nj)
            if _ppi != M:
                warn('not complete graph')

        self.check_inds(ii, jj, ij_ind)
        
        b = net.size(0)
        # net = net.view(b, ni, ppi, nj, DIMI).permute(0, 1, 3, 2, 4).reshape(-1, ppi, DIMI)
        # net = self.atten(net, net)
        # net = net.view(b, ni, nj, ppi, DIMI).permute(0, 1, 3, 2, 4).reshape(b, -1, DIMI)

        tokens = net[:, ij_ind, :].view(-1, M, DIMI)
        assert tokens.shape[0] == b * (len(ii) // M)

        if posenc_cache is None:
            warn('position encoding is NOT provided.')
            posenc_ = None
        else:
            # init shape: 2, b, 1, n_img*PPI, DIMI//8
            posenc_ = posenc_cache[:, :, :, ij_ind, :].view(2, tokens.shape[0], 1, M, DIMI//8)

        if self.version == 'a':
            tokens = self.atten(tokens, tokens)
        if self.version == 'pa':
            tokens = self.norm2(tokens)
            tem_t = tokens
            tokens = self.gatten(tokens, encoding=posenc_)
        else:
            raise NotImplementedError

        tokens = tokens.view(b, -1, DIMI)
        
        reverse_ind = torch.empty_like(ii)
        reverse_ind[ij_ind.view(-1)] = torch.arange(len(ii), device=ii.device)
        net = tokens[:, reverse_ind, :]

        return net, (self.d(net), self.w(net), None)
    
class SemanticPatchifier(nn.Module):
    def __init__(self, patch_size=3, compute_score=True, dino_adapt=True):
        super(SemanticPatchifier, self).__init__()
        self.patch_size = patch_size
        xfeat = load_model('verlab/accelerated_features', 'XFeat', 
                           verbose=True,
                           pretrained = True, top_k = 4096).net
        del xfeat.heatmap_head, xfeat.keypoint_head, xfeat.fine_matcher
        del xfeat.block3, xfeat.block4, xfeat.block5, xfeat.block_fusion
        self.xfeat = xfeat
        self.norm_f = nn.InstanceNorm2d(DIMF)
    
        self.dino = _make_dinov2_model(arch_name='vit_small', pretrained=True)
        if dino_adapt == True:
            self.dino_adapter = nn.Linear(DIMI, DIMI, bias=False)
        else:
            self.dino_adapter = nn.Identity()

        for param in self.dino.parameters():
            param.requires_grad = False

        if compute_score == True:
            self.score = nn.Sequential(nn.Linear(DIMI, DIMI // 3),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(DIMI // 3, 1),
                                    nn.Sigmoid())
        else:
            self.score = None

        self.pad_i = Padding(int(SCALEI), mode='right')


    def get_f(self, img):
        b, n, c1, H, W = img.shape
        img = img.view(b*n, c1, H, W)
        img = img * 2 - 0.5

        xfeat = self.xfeat
        with torch.no_grad():
            x = img.mean(dim=1, keepdim = True)
            x = xfeat.norm(x)

        x1 = xfeat.block1(x)
        x2 = xfeat.block2(x1 + xfeat.skip1(x))
        x2 = self.norm_f(x2)

        _, c2, h2, w2 = x2.shape
        return x2.view(b, n, c2, h2, w2)

    def get_i(self, img, seg_head=None, depth_head=None):
        b, n, c1, H, W = img.shape
        img = self.pad_i(img.view(b*n, c1, H, W))
        Hp, Wp = img.shape[-2:]

        # out_data = self.dino.forward_features(img)
        # tokens = out_data['x_norm_patchtokens'] # B, N, DIM
        # tokens_cls = out_data['x_norm_clstoken']

        x = self.dino._get_intermediate_layers_not_chunked(img,
                                                                n=[2,5,8,11],)
                                                                # reshape=False, # !
                                                                # return_class_token=True,
                                                                # norm=False, # !
                                                                # no_zip=True)

        x_norm = [self.dino.norm(tk) for tk in x]
        tokens_norm = x_norm[-1][:, 1:]

        c = tokens_norm.shape[-1]
        h, w = Hp // int(SCALEI), Wp // int(SCALEI)
        if self.score is not None:
            scores = self.score(tokens_norm)
            score_map = scores.reshape(b*n, h, w, 1).permute(0, 3, 1, 2).contiguous().view(b, n, 1, h, w)
        elif seg_head is not None:
            filt_out_mask = seg_head.get_filt(tokens_norm.view(-1, c))
            scores = (~filt_out_mask).half()
            score_map = scores.reshape(b*n, h, w, 1).permute(0, 3, 1, 2).contiguous().view(b, n, 1, h, w)
            score_map[..., 0, :] = 0.
            score_map[..., -1, :] = 0.
            score_map[..., 0] = 0.
            score_map[..., -1] = 0.
        else:
            scores, score_map = None, None

        if depth_head is not None:
            if isinstance(depth_head, DepthHeadFlatten):
                tokens_cls = x[-1][:, 0]
                tokens = x[-1][:, 1:]
                assert len(tokens.shape) == 3 and tokens.shape[0] == 1
                assert len(tokens_cls.shape) == 2 and tokens_cls.shape[0] == 1
                tokens = tokens[0]
                tokens_cls = tokens_cls.expand_as(tokens)
                tokens = torch.cat([tokens, tokens_cls], dim=-1)
                depth = depth_head(tokens) / depth_head.scale_factor
                depth_map = depth.reshape(b*n, h, w, 1).permute(0, 3, 1, 2).contiguous().view(b, n, h, w)
            else:
                tokens_cls_l = [v[:, 0] for v in x]
                tokens_l = [v[:, 1:] for v in x]
                tokens_map_l = [v.reshape(b, h, w, DIMI).permute(0, 3, 1, 2).contiguous()
                                for v in tokens_l]
                depth_map = depth_head(tuple(zip(tokens_map_l, tokens_cls_l))) / depth_head.scale_factor
                tokens_cls = tokens_cls_l[-1]
        else:
            depth_map, tokens_cls = None, None

        tokens_norm_map = tokens_norm.reshape(b*n, h, w, c).permute(0, 3, 1, 2).contiguous().view(b, n, c, h, w)
        out = {'tokens_norm_map': tokens_norm_map,
               'score_map': score_map,
               'depth_map': depth_map,
               'tokens_cls': tokens_cls
               }
        return out

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image, disps=None, 
                centroid_sel_strat='RANDOM_GRID', centroid_sample_strat='UPPERLEFT', return_color=False,
                seg_head=None, depth_head=None):
        """ extract patches from input images """
        # warn(f'centroid_sel_strat:{centroid_sel_strat}, centroid_sample_strat:{centroid_sample_strat},' )
        fmap = self.get_f(images) / 4.0
        iout = self.get_i(images, seg_head, depth_head)
        imap, score_map = [iout[key] for key in ['tokens_norm_map', 'score_map']]
        imap /= 4.0
        scores = None

        b, n, c, hf, wf = fmap.shape
        _, _, ci, hi, wi = imap.shape
        assert DIMF == c
        assert DIMI == ci
        if b > 1:
            warn('batch size > 1 !')

        scale_f2i = torch.tensor([SCALEF / SCALEI, SCALEF / SCALEI], device=imap.device)

        P = self.patch_size

        # bias patch selection towards regions with high gradient
        if centroid_sel_strat == 'GRADIENT_BIAS':
            g = self.__image_gradient(images)
            x = torch.randint(1, wf-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, hf-1, size=[n, 3*patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        if centroid_sel_strat == 'SCORE':
            # TODO: x, y >= R
            # warn('Using score to sample patches')
            assert score_map is not None # b, n, 1, h, w
            scores = score_map.view(b * n, hi * wi)
            hw_inds = torch.multinomial(scores, patches_per_image)
            scores = torch.gather(scores, 1, hw_inds).view(-1)

            y = torch.div(hw_inds, wi, rounding_mode='trunc')
            x = hw_inds - (y * wi)
            if centroid_sample_strat == 'UPPERLEFT':
                y = y.to(scale_f2i) / scale_f2i[0]
                x = x.to(scale_f2i) / scale_f2i[1]
            else:
                y = (y.to(scale_f2i) + 0.5) / scale_f2i[0]
                x = (x.to(scale_f2i) + 0.5) / scale_f2i[1]

        elif centroid_sel_strat == 'RANDOM':
            x = torch.randint(1, wf-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, hf-1, size=[n, patches_per_image], device="cuda")
        elif centroid_sel_strat == 'RANDOM_GRID':
            y, x = torch.meshgrid(
                    torch.arange(P//2, hi-P//2),
                    torch.arange(P//2, wi-P//2),
                    indexing = 'ij'
            )
            if centroid_sample_strat == 'UPPERLEFT':
                y = (y.to(scale_f2i) / scale_f2i[0]).reshape(-1)
                x = (x.to(scale_f2i) / scale_f2i[1]).reshape(-1)
            else:
                y = ((y.to(scale_f2i) + 0.5) / scale_f2i[0]).reshape(-1)
                x = ((x.to(scale_f2i) + 0.5) / scale_f2i[1]).reshape(-1)
            inds = torch.randperm(y.shape[0])[-patches_per_image:]
            y = y[inds].expand(n, -1)
            x = x[inds].expand(n, -1)

        else:
            raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")

        imode = 'bilinear' if centroid_sample_strat == 'BILINEAR' else 'upperleft'
        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], scale_f2i * coords, 0,   
                                mode=imode) #.view(b, -1, DIMI, 1, 1) # TODO: Only correct when b==1
        imap = self.dino_adapter(imap.view(b, -1, DIMI)).view(b, -1, DIMI, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords,             P//2).view(b, -1, DIMF, P, P) # b, patches_per_image, ...

        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            if iout['depth_map'] is not None:
                # warn('Using predicted depth for patch initialization')
                idisps = 1 / iout['depth_map']
                disps = F.interpolate(idisps, (hf, wf), mode='bilinear', align_corners=False)
            else:
                disps = torch.ones(b, n, hf, wf, device="cuda")
        else:
            assert disps.shape[-2:] == (hf, wf)

        grid, _ = coords_grid_with_index(disps, device=fmap.device) # grid: (b, n, 3, h, w), index: (b, n, 1, h, w)
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P) # -1: n * patches_per_image

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1) # (n, patches_per_image) -> n*p_p_i

        score_data = (score_map, scores) # -1: n * patches_per_image
        data_dict = {
            'score_data': score_data,
            'tokens_cls': iout['tokens_cls'],
        }
        if return_color:
            return fmap, gmap, imap, patches, index, clr, data_dict  

        return fmap, gmap, imap, patches, index, data_dict


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class CDVNet(nn.Module):
    def __init__(self, use_viewer=False, posenc=True, compute_score=False, dino_adapt=True):
        super(CDVNet, self).__init__()
        self.P = 3
        self.patchify = SemanticPatchifier(self.P, compute_score=compute_score, dino_adapt=dino_adapt)
        self.update = Update(self.P)

        self.DIM = DIMI
        self.DIMF = DIMF
        self.RES = 4
        self.setdense()

        head_dim = DIMI // 8
        if posenc:
            self.posenc = LearnableFourierPositionalEncoding(
                2, head_dim, head_dim
            )

        if compute_score == True:
            # ONLY train score
            for param in self.parameters():
                param.requires_grad = False
            for param in self.patchify.score.parameters():
                param.requires_grad = True

        trainable_params = [name for name, param in self.named_parameters() if param.requires_grad]

    @classmethod
    def add_edge(self, net, ii, jj, kk, ij_ind, ix, n, ppi):
        kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"), indexing='ij')
        kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"), indexing='ij')
        ii1, ii2 = ix[kk1], ix[kk2]

        ij_ind1 = torch.arange(n * ppi,                        device="cuda").view(n, ppi)
        ij_ind2 = torch.arange(n * ppi, n * ppi + ppi * (n+1), device="cuda").view(ppi, n+1).permute(1,0)

        n_new_edge = len(kk1) + len(kk2)
        ij_ind = ij_ind + n_new_edge

        ii = torch.cat([ii1, ii2, ii])
        jj = torch.cat([jj1, jj2, jj])
        kk = torch.cat([kk1, kk2, kk])
        ij_ind = torch.cat([ij_ind1, ij_ind2, ij_ind]) # (n_old, PPI) -> (n_new, PPI)

        net1 = torch.zeros(net.size(0), n_new_edge, DIMI, device="cuda")
        net = torch.cat([net1, net], dim=1)

        return net, ii, jj, kk, ij_ind
 
    @classmethod
    def edge_dropout(self, net, ii, jj, kk, ij_ind, n):
        if np.random.rand() < 0.1:
            k = (ii != (n - 4)) & (jj != (n - 4))
            ii = ii[k]
            jj = jj[k]
            kk = kk[k]
            net = net[:,k]

            ij_ind_keep = k[ij_ind] # (num_ij, PPI)
            ind_new = torch.cumsum(k.long(), dim=0) - 1
            assert (ij_ind_keep[:, :1] == ij_ind_keep).all()
            ij_ind = ind_new[ij_ind][ij_ind_keep[:, 0], :]
        return net, ii, jj, kk, ij_ind
    
    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """
        ppi = self.PPI

        images = images / 255.0
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        # patches:(b=1, n*patches_per_image, 3, P, P). ix: frame index of ALL patches
        fmap, gmap, imap, patches, ix, (score_map, scores) = self.patchify(images, disps=disps, patches_per_image=ppi)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        assert self.MIN_FRAME <= N

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        if hasattr(self, 'posenc'):
            p_shift = torch.tensor([w/2, h/2]).to(patches).view(1,1,2)
            p_scale = max(w/2, h/2)
            p_n = (patches[:, :, :2, p//2, p//2] - p_shift) / p_scale
            posenc_cache = self.posenc(p_n) # 2, b, 1, n_img*PPI, DIMI//8
            getcache = lambda kk : posenc_cache[:,:,:,kk]
        else:
            getcache = lambda kk : None

        kk, jj = flatmeshgrid(torch.where(ix < self.MIN_FRAME)[0], torch.arange(0,self.MIN_FRAME, device="cuda"), indexing='ij')
        ii = ix[kk]

        ni, nj = len(torch.unique(ii)), len(torch.unique(jj))
        # ij_ind : index such that ii[ij_ind] and jj[ij_ind] are contiguous
        ij_ind = torch.arange(ni*ppi*nj, device="cuda").view(ni, ppi, nj).permute(0, 2, 1).reshape(-1, ppi)

        imap = imap.view(b, -1, DIMI)
        net = torch.zeros(b, len(kk), DIMI, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= self.MIN_FRAME and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]

                net, ii, jj, kk, ij_ind = self.add_edge(net, ii, jj, kk, ij_ind, ix, n, ppi)

                net, ii, jj, kk, ij_ind = self.edge_dropout(net, ii, jj, kk, ij_ind, n)

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk) # iproj make patches(b,-1,3,P,P) to X(b,-1,P,P,4)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, 
                                                  ii, jj, kk, ij_ind, posenc_cache=getcache(kk)) # posenc_cache[:,:,:,kk] !!!!
            if self.patchify.score is not None:
                s_kk = scores[kk]
                weight = weight * s_kk.view(1, -1, 1)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            if score_map is not None:
                scores_out = scores[kk[k]]
            else:
                scores_out = None
            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], (score_map, scores_out)))

        return traj

    def setsparse(self):
        self.PPI = 80
        self.MIN_FRAME = 8

    def setdense(self):
        self.PPI = 1530# 1530 # 1610 # 80 # Patches Per Image
        self.MIN_FRAME = 4 # 8

class SegHeadFlatten(nn.Module):
    def __init__(self, in_channels=384, num_classes=21):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes

        self.bn = nn.BatchNorm1d(self.in_channels) # The Name should match the setting of mmcv
        self.linear = nn.Linear(self.in_channels, self.out_channels) # The Name should match the setting of mmcv

        self.filt_index = None

    def forward(self, x):
        # x:(b,c)
        x = self.bn(x)
        out = self.linear(x)
        return out
    
        
    def get_filt(self, x):
        seg_logit = self.forward(x)
        seg_class = torch.argmax(seg_logit, dim=-1, keepdim=False)
        seg_class = seg_class.to(torch.uint8)
        filt_mask = torch.isin(seg_class, self.filt_index)
        return filt_mask

class DepthHeadFlatten(nn.Module):
    def __init__(self, in_channels=768, n_bins=256):
        super().__init__()
        self.in_channels = in_channels
        self.n_bins = n_bins
        self.min_depth=0.001
        self.max_depth=80
        self.linear = nn.Linear(self.in_channels, self.n_bins) # The Name should match the setting of mmcv
    def forward(self, x):
        # x: (b, c)
        logit = self.linear(x)

        # bins_strategy == "UD":
        bins = torch.linspace(self.min_depth, self.max_depth, self.n_bins, device=x.device)


        # following Adabins, default linear
        # if self.norm_strategy == "linear":
        logit = torch.relu(logit)
        eps = 0.1
        logit = logit + eps
        logit = logit / logit.sum(dim=1, keepdim=True)

        output = torch.matmul(logit, bins[:, None])
        # output = torch.einsum("bnc,c->bn", [logit, bins]).unsqueeze(dim=2)
        return output
