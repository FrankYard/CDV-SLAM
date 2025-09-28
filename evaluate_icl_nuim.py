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
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--iclnuim_dir', default="datasets/ICL_NUIM", type=Path)
    parser.add_argument('--backend_thresh', type=float, default=64.0)
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

    scenes = [
        "living_room_traj0_loop",
        "living_room_traj1_loop",
        "living_room_traj2_loop",
        "living_room_traj3_loop",
        "office_room_traj0_loop",
        "office_room_traj1_loop",
        "office_room_traj2_loop",
        "office_room_traj3_loop",
    ]

    net = get_net(args.version, args.network)

    results = {}
    for scene in scenes:
        imagedir = args.iclnuim_dir / scene
        if scene.startswith("living"):
            groundtruth = args.iclnuim_dir / f"TrajectoryGT" / f"livingRoom{scene[-6]}.gt.freiburg"
        elif scene.startswith("office"):
            groundtruth = args.iclnuim_dir / f"TrajectoryGT" / f"traj{scene[-6]}.gt.freiburg"
        traj_ref = file_interface.read_tum_trajectory_file(groundtruth)

        scene_results = []
        for i in range(args.trials):
            traj_est, timestamps = run(cfg, net, imagedir, "calib/icl_nuim.txt", args.stride, args.viz, args.show_img)

            images_list = sorted(glob.glob(os.path.join(imagedir, "*.png")))[::args.stride]
            tstamps = np.arange(1, len(images_list)+1, args.stride, dtype=np.float64)

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=tstamps)

            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

            result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
            ate_score = result.stats["rmse"]

            if args.plot:
                scene_name = scene.rstrip("_loop").title()
                Path("trajectory_plots").mkdir(exist_ok=True)
                plot_trajectory(traj_est, traj_ref, f"ICL_NUIM {scene_name} Trial #{i+1} (ATE: {ate_score:.03f})",
                                f"trajectory_plots/ICL_NUIM_{scene_name}_Trial{i+1:02d}.pdf", align=True, correct_scale=True)

            if args.save_trajectory:
                Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/ICL_NUIM_{scene_name}_Trial{i+1:02d}.txt", traj_est)

            scene_results.append(ate_score)

        results[scene] = np.median(scene_results)
        print(scene, sorted(scene_results))

    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(results[scene])

    file_path = os.path.join('results', f"icl_nuim_{args.expname}.txt")
    with open(file_path, 'a') as f:
        print(args.network, '\n', file=f)
        for k in results:
            print(k, results[k], file=f)
        print("AVG: ", np.mean(xs), file=f)
        print('\n', file=f)

    print("AVG: ", np.mean(xs))
