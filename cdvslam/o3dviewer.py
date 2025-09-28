import numpy as np
import torch
import cv2
import open3d as o3d
from collections import OrderedDict, defaultdict
from .lietorch import SE3
from torch.multiprocessing import Process, Value, Queue

from matplotlib import colormaps as img_colormaps
import os

import DINO_modules.datamaps as colormaps

# import droid_backends
RANGE = 50. #200
NUM_FLUSH = 10#64
FLUSH_CONTINUE = False
NUM_TRACK = 5
FOLLOW = False 
FOLLOW_TOP = False
SCREEN_SHOT = False
VISIBLE = True
CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

# colormap_array = np.array(colormaps.VOC2012_COLORMAP, dtype=np.uint8)
COLORMAP_ARRAY = np.array(colormaps.ADE20K_COLORMAP, dtype=np.uint8)[1:]
TEXTS = [t.split(';')[0] for t in colormaps.ADE20K_CLASS_NAMES][1:]

def look_at(eye, center, up):
    forward = center - eye
    forward = forward / np.linalg.norm(forward)

    # x-axis of camera in world coordinate
    right = np.cross(forward, up)
    assert np.linalg.norm(right) > 1e-6
    right = right / np.linalg.norm(right)

    # y-axis
    new_up = np.cross(right, forward)
    
    # extrinsic: camera to world
    # [ R | T ]
    # [ 0 | 1 ]
    extrinsic = np.identity(4)
    extrinsic[0:3, 0] = right
    extrinsic[0:3, 1] = -new_up
    extrinsic[0:3, 2] = forward
    extrinsic[0:3, 3] = eye
    
    return extrinsic

class O3DViewer:
    def __init__(self, image, poses, points, colors, intrinsics, dirty, weight, seg, 
                 tcounts, frame_n,
                 device="cuda:0", points_frame_capacity=512, warmup=1, filter_thresh=0.005, weight_stage=-3):
        torch.cuda.set_device(device)
        self.image_buffer = image
        self.poses_buffer = poses
        self.coords_buffer = points
        self.colors_buffer = colors
        self.intrinsics_buffer = intrinsics
        self.dirty = dirty
        self.weight = weight
        self.seg = seg
        self.tcounts = tcounts
        self.frame_n = frame_n
        self.delta_queue = Queue()
        self.flush_index_start = None

        self.cameras = {} # keyframe only
        self.nonkey_cameras = {}
        self.frame_id_2_keyframe_id = {}
        self.show_nonkeyframe = True
        self.points = OrderedDict()
        self.frame_deltas = defaultdict(dict)
        self.points_frame_capacity = points_frame_capacity
        self.warmup = warmup
        self.warmupped = False
        self.filter_thresh = filter_thresh
        self.weight_stage = weight_stage
        self.weight_thresh_ = 0.125
        self.ix = 0
        self.num_image = Value('i', 0)
        self.image_queue = Queue()
        self.follow_view = FOLLOW
        self.mode = 'seg' # 'seg'
        self.vis = None

        self.process = Process(target=self.run)
        self.process.start()
        
    def _create_camera_actor(self, rgb=(0.5, 0.5, 1.), scale=0.05):
        """ build open3d camera polydata """
        camera_actor = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
            lines=o3d.utility.Vector2iVector(CAM_LINES))

        # color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
        color = rgb
        camera_actor.paint_uniform_color(color)
        return camera_actor

    def _create_point_actor(self, points, colors):
        """ open3d point cloud from numpy array """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud

    def _increase_filter(self, vis):
        self.filter_thresh *= 2
        with self.frame_n.get_lock():
            self.dirty[:self.frame_n.value] = True

    def _decrease_filter(self, vis):
        self.filter_thresh *= 0.5
        # with self.video.get_lock():
        #     self.video.dirty[:self.video.counter.value] = True

    def _increase_weight(self, vis):
        self.weight_stage += 1
        self.update_weight_thresh()
        with self.frame_n.get_lock():
            self.dirty[:self.frame_n.value] = True

    def _decrease_weight(self, vis):
        self.weight_stage -= 1
        self.update_weight_thresh()
        with self.frame_n.get_lock():
            self.dirty[:self.frame_n.value] = True

    def update_weight_thresh(self):
        if self.weight_stage == 0:
            self.weight_thresh_ = 0.5
        elif self.weight_stage < 0:
            self.weight_thresh_ = 2**(self.weight_stage)
        else:
            self.weight_thresh_ = 1 - 2**(-self.weight_stage)
        print('weight thresh updated:', self.weight_thresh_)

    def get_color(self, dirty_index):
        mask = True
        if self.seg is None or self.mode == 'img':
            Colors = self.colors_buffer[dirty_index].cpu().numpy() / 255.0
        elif self.mode == 'seg_no_back':
            seg_class = self.seg[dirty_index].cpu().numpy()
            Colors = COLORMAP_ARRAY[seg_class] / 255.0
            mask = (seg_class != 0)
        else:
            seg_class = self.seg[dirty_index].cpu().numpy()
            Colors = COLORMAP_ARRAY[seg_class] / 255.0
        return Colors, mask


    def _dump_pose(self, vis):
        pass
        # self.video.dump_pose()

    def _set_view(self, vis):
        if self.follow_view == True:
            self.follow_view = False
        else:
            self.follow_view = True

    def _set_O(self, vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        cam.extrinsic = np.eye(4)
        vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

    def _switch_mode(self, vis):
        if self.seg is None:
            return
        if self.mode == 'img':
            self.mode = 'seg'
        elif self.mode == 'seg':
            self.mode = 'seg_no_back'
        else:
            self.mode = 'img'
        with self.frame_n.get_lock():
            self.dirty[:self.frame_n.value] = True
        print('vis mode:', self.mode)

    def max_frame_id(self):
        max_keyframe_id = self.tcounts[max(self.cameras, default=0)].item()
        return max(max_keyframe_id, max(self.nonkey_cameras, default=0))

    def _screen_shot(self, i=None, nonkey=False, tstamp=''):
        if SCREEN_SHOT:
            if FOLLOW_TOP:
                cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
                last_cam_i = len(self.cameras)-1
                cam.extrinsic = SE3(self.poses_buffer[last_cam_i]).matrix().cpu().numpy()
                center = np.linalg.inv(cam.extrinsic)[:3, 3]
                eye = center + np.array([0, -10, 0])
                up = np.array([0, 0, 1])
                cam.extrinsic = np.linalg.inv(look_at(eye, center, up))
                self.vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

            self.vis.update_renderer()
            if i is None:
                i = self.max_frame_id()
            nonkey = 'nk' if nonkey else ''
            self.vis.capture_screen_image(f"o3d_capture/image_{i:04d}.png", do_render=True) # _{nonkey}{tstamp}

    def add_delta(self, t, t0, dP, kid=None):
        dP_mat = dP.matrix().cpu().numpy()
        with self.frame_n.get_lock():
            self.delta_queue.put_nowait((t, t0, dP_mat, kid))

    def recieve_delta(self):
        with self.frame_n.get_lock():
            while not self.delta_queue.empty():
                t, t0, dP_mat, keyframe_id = self.delta_queue.get()
                if keyframe_id is not None:
                    self.frame_id_2_keyframe_id[t0] = keyframe_id
                else:
                    if t0 in self.frame_id_2_keyframe_id:
                        keyframe_id = self.frame_id_2_keyframe_id[t0]
                    else:
                        print(f'missed keyframe index:{t}->{t0}')
                        break
                self.frame_deltas[keyframe_id][t] = dP_mat

    def add_nonkeyframe(self, keyframe_i, keyfrmae_pose_mat):
        if keyframe_i in self.frame_deltas:
            for t, dp in self.frame_deltas[keyframe_i].items():
                if t in self.nonkey_cameras:
                    self.vis.remove_geometry(self.nonkey_cameras[t])
                    del self.nonkey_cameras[t]
                if self.show_nonkeyframe:
                    pose = np.matmul(keyfrmae_pose_mat, np.linalg.inv(dp))
                    cam_actor = self._create_camera_actor()
                    cam_actor.transform(pose)
                    self.vis.add_geometry(cam_actor)
                    self.nonkey_cameras[t] = cam_actor

                    if hasattr(self, 'gt') and not t in self.gt_shown:
                        self.vis.add_geometry(self.gt[t])
                        if hasattr(self, 'dpv'):
                            self.vis.add_geometry(self.dpv[t])
                        self.gt_shown.add(t)
                        self._screen_shot(nonkey=True, tstamp=t)


    def _switch_nonkeyframe(self, vis):
        self.show_nonkeyframe = False if self.show_nonkeyframe else True
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        for t0, dict_t0 in self.frame_deltas.items():
            if self.show_nonkeyframe:
                keyfrmae_pose_mat = SE3(self.poses_buffer[t0]).inv().matrix().cpu().numpy()
            for t, dp in dict_t0.items():
                if t in self.nonkey_cameras:
                    vis.remove_geometry(self.nonkey_cameras[t])
                    del self.nonkey_cameras[t]
                if self.show_nonkeyframe:
                    pose = np.matmul(keyfrmae_pose_mat, np.linalg.inv(dp))
                    cam_actor = self._create_camera_actor()
                    cam_actor.transform(pose)
                    vis.add_geometry(cam_actor)
                    self.nonkey_cameras[t] = cam_actor     

        view_control = vis.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)       

    def  _animation_callback(self, vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        with torch.no_grad():
            with self.frame_n.get_lock():
                dirty_index , = torch.where(self.dirty.clone())
                num_keyframe = self.frame_n.value
            if len(dirty_index) == 0:
                return

            if NUM_FLUSH is not None:
                if self.flush_index_start is not None and FLUSH_CONTINUE:
                    dirty_start = torch.where(dirty_index == self.flush_index_start)[0][0].item()
                else:
                    dirty_start = 0
                dirty_trunc = dirty_start + NUM_FLUSH - NUM_TRACK
                if dirty_start + NUM_FLUSH < len(dirty_index):
                    self.flush_index_start = dirty_index[dirty_trunc]
                    dirty_index = torch.cat([dirty_index[dirty_start: dirty_trunc], 
                                             dirty_index[len(dirty_index)-NUM_TRACK:] ])
                else:
                    self.flush_index_start = None
                    dirty_index = dirty_index[dirty_start : ]

            assert len(dirty_index) <= NUM_FLUSH
            with self.frame_n.get_lock():
                self.dirty[dirty_index] = False

                Poses = SE3(self.poses_buffer[dirty_index]).inv().matrix().cpu().numpy()
                Coords = self.coords_buffer[dirty_index].cpu().numpy() # num_image, M, 3
                Colors, keep_mask = self.get_color(dirty_index) # Color range: [0, 1]
                weight = self.weight[dirty_index].cpu().numpy()
                tcounts = self.tcounts[dirty_index].cpu().numpy()
                
            mask = keep_mask * (Coords < RANGE).all(-1) * (Coords > -RANGE).all(-1)
            mask *= (weight > self.weight_thresh_)
            for i, ix in enumerate(sorted(dirty_index.tolist())):
                pose = Poses[i]
                point = Coords[i]
                color  = Colors[i]
                tstamp = tcounts[i].item()

                self.frame_id_2_keyframe_id[tstamp] = ix

                m = mask[i]
                if ix in self.cameras:
                    vis.remove_geometry(self.cameras[ix])
                    del self.cameras[ix]
                cam_actor = self._create_camera_actor()
                cam_actor.transform(pose)
                vis.add_geometry(cam_actor)
                self.cameras[ix] = cam_actor
                
                if hasattr(self, 'gt') and not tstamp in self.gt_shown:
                    self.vis.add_geometry(self.gt[tstamp])
                    if hasattr(self, 'dpv'):
                        self.vis.add_geometry(self.dpv[tstamp])
                    self.gt_shown.add(tstamp)
                self._screen_shot(tstamp=tstamp)

                self.add_nonkeyframe(ix, pose)

                if ix in self.points:
                    if m.any():
                        point_actor = self.points[ix]
                        point_actor.points = o3d.utility.Vector3dVector(point[m])
                        point_actor.colors = o3d.utility.Vector3dVector(color[m])
                        vis.update_geometry(point_actor)
                    else:
                        vis.remove_geometry(self.points[ix])
                        del self.points[ix]
                elif m.any():
                    point_actor = self._create_point_actor(point[m], color[m])
                    vis.add_geometry(point_actor)
                    self.points[ix] = point_actor

        if self.follow_view:
            last_cam_i = len(self.cameras)-1
            cam.extrinsic = SE3(self.poses_buffer[last_cam_i]).matrix().cpu().numpy()
            if FOLLOW_TOP:
                center = np.linalg.inv(cam.extrinsic)[:3, 3]
                eye = center + np.array([0, -10, 0])
                up = np.array([0, 0, 1])
                cam.extrinsic = np.linalg.inv(look_at(eye, center, up))

            self.follow_view = FOLLOW

        if FOLLOW_TOP or (len(self.cameras) >= self.warmup and self.warmupped):
            view_control = vis.get_view_control()
            view_control.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

        if not self.warmupped:
            self.warmupped = True

        self.recieve_delta()

        vis.poll_events()
        vis.update_renderer()

    def update_image(self, image, patches=None, weights=None, seg_class=None, scale=None, save_dir=None,seg_text=True):
        image = image.permute(1, 2, 0).cpu().numpy() / 255.0
        if seg_class is not None:
            seg_class = seg_class.cpu().numpy() # shape == (M,)
            class_color = COLORMAP_ARRAY[seg_class] / 255.0
            if seg_text:
                text_img = np.zeros(image.shape, dtype=np.uint8)

        if patches is not None:
            patches = patches.cpu().numpy()
            weights = weights.cpu().numpy()
            weights = weights / weights.max()
            s = 1 if scale is None else scale
            for i, (ptc, w) in enumerate(zip(patches, weights)):
                p0, p1 = ptc[:, 0, 0], ptc[:, -1, -1]
                x0, x1, y0, y1 = [int(val * s) for val in [p0[0], p1[0], p0[1], p1[1]]]
                w_color = np.array(img_colormaps['viridis'](w)).reshape(1,1,4)

                image[y0:y1+1, x0:x1+1] = (image[y0:y1+1, x0:x1+1] + w_color[:,:,:3] * 2) / 3
                if seg_class is not None:
                    if seg_class[i] != 0:
                        yc, xc  = (y0+y1)//2, (x0+x1)//2
                        image[yc-1: yc+2, xc-1:xc+2] = class_color[i]
                    if seg_text:
                        text = TEXTS[seg_class[i]]
                        text_img = cv2.putText(text_img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 
                                               0.5, [int(cl) for cl in class_color[i]*255], 1)

        if seg_class is not None and seg_text:
            image = np.where(text_img, text_img/255.0, image)
        cv2.imshow('image', image)
        cv2.waitKey(2)

        if SCREEN_SHOT:
            save_dir = 'ocv_capture'
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, f'image_{self.num_image.value:04d}.png'), (image * 255).astype(np.uint8))
        with self.num_image.get_lock():
            self.num_image.value += 1

    def run(self, height=720, width=1280, render_option="misc/renderoption.json"):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_animation_callback(self._animation_callback)
        self.vis.register_key_callback(ord("S"), self._increase_filter)
        self.vis.register_key_callback(ord("A"), self._decrease_filter)
        self.vis.register_key_callback(ord("Z"), self._increase_weight)
        self.vis.register_key_callback(ord("X"), self._decrease_weight)

        self.vis.register_key_callback(ord("D"), self._dump_pose)
        self.vis.register_key_callback(ord("F"), self._set_view)
        self.vis.register_key_callback(ord("O"), self._set_O)
        self.vis.register_key_callback(ord("N"), self._switch_nonkeyframe)

        self.vis.register_key_callback(ord("]"), self._switch_mode)
        self.vis.create_window(height=height, width=width, visible=VISIBLE)
        print('window created')
        if render_option:
            self.vis.get_render_option().load_from_json(render_option)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0,
            origin=[0, 0, 0] 
        )
        self.vis.add_geometry(coord_frame)

        if FOLLOW_TOP:
            center = np.array([0, 0, 0])
            eye = center + np.array([0, -10, 0])
            up = np.array([0, 0, 1])
            cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam.extrinsic = np.linalg.inv(look_at(eye, center, up))
            view_control = self.vis.get_view_control()
            view_control.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

            opt = self.vis.get_render_option() 
            opt.background_color = np.asarray([0, 0, 0])
            
            self.vis.poll_events()
            self.vis.update_renderer()

        self.vis.run()
        self.vis.destroy_window()
        print('Vis finished')

    def join(self):
        self.process.join()