import cv2
import os
import argparse
import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from cdvslam.data_readers.factory import dataset_factory

from cdvslam.lietorch import SE3
from cdvslam.logger import Logger
import torch.nn.functional as F

from cdvslam.net_dpv import VONet
from evaluate_tartan import evaluate as validate

from tqdm import tqdm
# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, args):
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=rank)

    torch.manual_seed(0)
    torch.cuda.set_device(rank)

def load_pretrained_offical(net, ckpt_path):
    state_dict = torch.load(ckpt_path)
    new_state_dict = OrderedDict()
    if isinstance(net, DDP):
        for k, v in state_dict.items():
            if k.startswith('module.update.lmbda'):
                continue
            new_state_dict[k] = v
    else:
        for k, v in state_dict.items():
            if k.startswith('module.update.lmbda'):
                continue
            new_state_dict[k.replace('module.', '')] = v
    net.load_state_dict(new_state_dict, strict=True)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c


def train(rank, args):
    """ main training loop """

    # legacy ddp code
    setup_ddp(rank, args)


    db = dataset_factory(['tartan'], datapath="datasets/TartanAir", n_frames=args.n_frames)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(db, batch_size=1, sampler=train_sampler, num_workers=2)

    net = VONet()
    net.train()
    net.cuda()

    ddpnet = DDP(net, device_ids=[rank], find_unused_parameters=False)

    if args.ckpt is not None:
        load_pretrained_offical(ddpnet, args.ckpt)

    optimizer = torch.optim.AdamW(ddpnet.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    if rank == 0:
        logger = Logger(args.name, scheduler)
        if args.ckpt is not None:
            validation_results = validate(None, ddpnet.module)
            logger.write_dict(validation_results)

    total_steps = 0
    epoch = 0
    while 1:
        epoch += 1
        for data_blob in tqdm(train_loader):
            images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
            optimizer.zero_grad()

            # fix poses to gt for first 1k steps
            so = total_steps < 1000 and args.ckpt is None

            poses = SE3(poses).inv()
            traj = ddpnet(images, poses, disps, intrinsics, M=1024, STEPS=18, structure_only=so)

            loss = 0.0
            for i, (v, x, y, P1, P2, kl) in enumerate(traj):
                e = (x - y).norm(dim=-1)
                e = e.reshape(-1, ddpnet.module.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values

                N = P1.shape[1]
                ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
                ii = ii.reshape(-1).cuda()
                jj = jj.reshape(-1).cuda()

                k = ii != jj
                ii = ii[k]
                jj = jj[k]

                P1 = P1.inv()
                P2 = P2.inv()

                t1 = P1.matrix()[...,:3,3]
                t2 = P2.matrix()[...,:3,3]

                s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                P1 = P1.scale(s.view(1, 1))

                dP = P1[:,ii].inv() * P1[:,jj]
                dG = P2[:,ii].inv() * P2[:,jj]

                e1 = (dP * dG.inv()).log()
                tr = e1[...,0:3].norm(dim=-1)
                ro = e1[...,3:6].norm(dim=-1)

                loss += args.flow_weight * e.mean()
                if not so and i >= 2:
                    loss += args.pose_weight * ( tr.mean() + ro.mean() )

            # kl is 0 (not longer used)
            loss += kl
            loss.backward()

            torch.nn.utils.clip_grad_norm_(ddpnet.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            total_steps += 1

            metrics = {
                "epoch": epoch,
                "loss": loss.item(),
                "kl": kl.item(),
                "px1": (e < .25).float().mean().item(),
                "ro": ro.float().mean().item(),
                "tr": tr.float().mean().item(),
                "r1": (ro < .001).float().mean().item(),
                "r2": (ro < .01).float().mean().item(),
                "t1": (tr < .001).float().mean().item(),
                "t2": (tr < .01).float().mean().item(),
            }

            if rank == 0:
                logger.push(metrics)

            if total_steps % 10000 == 0 or total_steps == 2:
                torch.cuda.empty_cache()

                if rank == 0:
                    PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                    if not os.path.isdir('checkpoints'):
                        os.mkdir('checkpoints')
                    torch.save(ddpnet.state_dict(), PATH)

                    validation_results = validate(None, ddpnet.module)
                if rank == 0:
                    logger.write_dict(validation_results)

                torch.cuda.empty_cache()
                ddpnet.train()

    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    args.world_size = args.gpus
    print(args)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train, nprocs=args.world_size, args=(args,))
