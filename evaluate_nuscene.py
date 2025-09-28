from itertools import count
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot
import json
import os
from tqdm import tqdm

from cdvslam.config import cfg
from cdvslam.slam import SLAM
from cdvslam.plot_utils import plot_trajectory
from cdvslam.utils import Timer, get_net

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def check_nuscene_info(nuscenedir, sequence, occ_vo_gt_dir):
    occ_gt_file = occ_vo_gt_dir / f"{sequence}.txt"
    nuscene_file = nuscenedir / "voxel04" / "annotations.json"
    occ_ts = []
    occ_trans = []
    occ_rots = []
    with open(occ_gt_file, 'r') as f:
        for line in f.readlines():
            fields = line.strip('\n').split(' ')
            occ_ts.append(int(float(fields[0]) * 1e6))
            occ_trans.append(fields[1:4])
            occ_rots.append(fields[7:] + fields[4:6])

    with open(nuscene_file, 'r') as f:
        annotations = json.load(f)
    scene_infos = annotations['scene_infos']
    scene_info = scene_infos[sequence]

    for i, (frame_token, frame_data) in enumerate(scene_info.items()):

        t = frame_data['camera_sensor']['CAM_FRONT']['ego_pose']['timestamp']
        rotation = frame_data['camera_sensor']['CAM_FRONT']['ego_pose']['rotation']
        translation = frame_data['camera_sensor']['CAM_FRONT']['ego_pose']['translation']
        occ_t, occ_tran, occ_rot = occ_ts[i], occ_trans[i], occ_rots[i]
        assert occ_t == int(t), (occ_t, int(t))
        for r1, r2 in zip(occ_rot, rotation):
            assert float(r1) == r2, (r1, r2)
        for tr1, tr2 in zip(occ_tran, translation):
            assert float(tr1) == tr2
    return 'succsess'

def nuscene_image_stream(queue, nuscenedir, sequence, stride, skip=0):
    with open(nuscenedir / "voxel04" / "annotations.json", 'r') as f:
        annotations = json.load(f)
    scene_infos = annotations['scene_infos']
    scene_info = scene_infos[sequence]

    for frame_token, frame_data in scene_info.items():

        t = frame_data['camera_sensor']['CAM_FRONT']['ego_pose']['timestamp']
        img_path = frame_data['camera_sensor']['CAM_FRONT']['img_path']
        intrinsics = frame_data['camera_sensor']['CAM_FRONT']['intrinsics']

        t /= 1e6
 
        intrinsics = np.array(intrinsics)[[0, 1, 0, 1], [0, 1, 2, 2]]

        image = cv2.imread(str(nuscenedir / "voxel04" / 'imgs' / img_path))
        
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        intrinsics /= 2

        H, W, _ = image.shape
        H, W = (H - H%4, W - W%4)
        image = image[..., :H, :W, :]

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


@torch.no_grad()
def run(cfg, network, nuscenedir, sequence, stride=1, viz=False, show_img=False):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=nuscene_image_stream, args=(queue, nuscenedir, sequence, stride, 0))
    reader.start()

    for step in count(start=1):
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.as_tensor(image, device='cuda').permute(2,0,1)
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')

        if show_img:
            show_image(image, 1)

        if slam is None:
            slam = SLAM(cfg, network, ht=image.shape[-2], wd=image.shape[-1], viz=viz)

        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=False):
            out = slam(t, image, intrinsics)

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
    parser.add_argument('--nuscene_dir', type=Path, default="datasets/NuScene") 
    parser.add_argument('--occvo_dir', type=Path, default='datasets/OCCVO')

    parser.add_argument('--backend_thresh', type=float, default=32.0)
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

    nuscene_scenes = sorted(os.listdir('Occ3D-nuScenes/OCCVO/tpvformer_occupancy_data/tpv'))

    for scene in nuscene_scenes:
        check_nuscene_info(Path('NuScene'),
                            scene,
                            Path('Occ3D-nuScenes/OCCVO/trajectory_groundtruth/gt_sample'))
        print(scene, 'data passed check')

    net = get_net(args.version, args.network) 

    results = {}
    for scene in nuscene_scenes:
        groundtruth = args.occvo_dir / 'trajectory_groundtruth'/ 'gt_sample' / f"{scene}.kitti"
        poses_ref = file_interface.read_kitti_poses_file(groundtruth)
        print(f"Evaluating OccVo {scene} with {poses_ref.num_poses // args.stride} poses")

        scene_results = []
        for trial_num in range(args.trials):
            traj_est, timestamps = run(cfg, net, args.nuscene_dir, scene, args.stride, args.viz, args.show_img)

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=np.arange(poses_ref.num_poses, dtype=np.float64)) # use integer Time stamp !!!

            traj_ref = PoseTrajectory3D(
                positions_xyz=poses_ref.positions_xyz,
                orientations_quat_wxyz=poses_ref.orientations_quat_wxyz,
                timestamps=np.arange(poses_ref.num_poses, dtype=np.float64))

            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

            try:
                result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                    pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
                ate_score = result.stats["rmse"]
                align = True
            except Exception as e:
                print(e)
                ate_score = np.inf
                align = False

            if args.plot:
                Path("trajectory_plots").mkdir(exist_ok=True)
                plot_trajectory(traj_est, traj_ref, f"nuscene sequence {scene} Trial #{trial_num+1}", 
                                f"trajectory_plots/nuscene_seq{scene}_trial{trial_num+1:02d}.pdf", 
                                align=align, correct_scale=True, plot_mode=plot.PlotMode.xy)

            if args.save_trajectory:
                Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/nuscene_{scene}.txt", traj_est)

            scene_results.append(ate_score)

        results[scene] = sorted(scene_results)
        print(scene, sorted(scene_results))

    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(np.median(results[scene]))

    import os
    file_path = os.path.join('results', f"nuscene_{args.expname}.txt")
    with open(file_path, 'a') as f:
        print(args.network, '\n', file=f)
        for k in results:
            print(k, results[k], file=f)
        print("AVG: ", np.mean(xs), file=f)
        print('\n', file=f)

    print("AVG: ", np.mean(xs))
