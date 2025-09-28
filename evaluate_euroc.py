import glob
import os
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

import psutil

from cdvslam.config import cfg
from cdvslam.slam import SLAM
from cdvslam.plot_utils import plot_trajectory
from cdvslam.stream import image_stream
from cdvslam.utils import Timer, get_net

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, viz=False, show_img=False):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, 0))
    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if show_img:
            show_image(image, 1)

        if slam is None:
            slam = SLAM(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)

    reader.join()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="cdv")
    parser.add_argument('--expname', default='')

    parser.add_argument('--network', type=str, default='cdv_dinov2.pth')
    parser.add_argument('--config', default="config/default_cdvo.yaml")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--eurocdir', default="datasets/EUROC")
    parser.add_argument('--backend_thresh', type=float, default=96.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    euroc_scenes = [
        "MH_01_easy",
        "MH_02_easy",
        "MH_03_medium",
        "MH_04_difficult",
        "MH_05_difficult",
        "V1_01_easy",
        "V1_02_medium",
        "V1_03_difficult",
        "V2_01_easy",
        "V2_02_medium",
        "V2_03_difficult",
    ]
    transes = []

    net = get_net(args.version, args.network) 

    results = {}
    for scene in euroc_scenes:
        imagedir = os.path.join(args.eurocdir, scene, "mav0/cam0/data")
        groundtruth = "datasets/euroc_groundtruth/{}.txt".format(scene) 

        scene_results = []
        for i in range(args.trials):
            traj_est, timestamps = run(cfg, net, imagedir, "calib/euroc.txt", args.stride, args.viz, args.show_img)

            images_list = sorted(glob.glob(os.path.join(imagedir, "*.png")))[::args.stride]
            tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=np.array(tstamps))

            traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

            result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
            transes.append(result.np_arrays['alignment_transformation_sim3'])
            ate_score = result.stats["rmse"]

            if args.plot:
                scene_name = '_'.join(scene.split('/')[1:]).title()
                Path("trajectory_plots").mkdir(exist_ok=True)
                plot_trajectory(traj_est, traj_ref, f"Euroc {scene} Trial #{i+1} (ATE: {ate_score:.03f})",
                                f"trajectory_plots/Euroc_{scene}_Trial{i+1:02d}.pdf", align=True, correct_scale=True)

            if args.save_trajectory:
                Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/Euroc_{scene}_Trial{i+1:02d}.txt", traj_est)

            scene_results.append(ate_score)
            print('ATE:', ate_score)
        results[scene] = sorted(scene_results)
        print(scene, sorted(scene_results))
        torch.save(np.stack(transes, axis=0), 'transes_euroc')

    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(np.median(results[scene]))

    file_path = os.path.join('results', f"euroc_{args.expname}.txt")
    with open(file_path, 'a') as f:
        print(args.network, '\n', file=f)
        

        current_pid = os.getpid()
        current_process = psutil.Process(current_pid)
        full_cmd = current_process.cmdline()
        full_command = ' '.join(full_cmd)
        print(full_command, '\n', file=f)

        for k in results:
            print(k, results[k], file=f)
        print("AVG: ", np.mean(xs), file=f)
        print('\n', file=f)

    print("AVG: ", np.mean(xs))

    

    
