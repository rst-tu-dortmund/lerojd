import copy
import pickle
import os

import numpy as np
from skimage import io

import random
# from pcdet.datasets.VoD_lidar_radar.VoD_utils_lidar import gen_sparse_points
import json
import os
from pathlib import Path

from . import VoD_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

from ..VoD_lidar_radar.vod.configuration.file_locations import KittiLocations
from ..VoD_lidar_radar.vod.frame import FrameDataLoader
from ..VoD_lidar_radar.vod.frame import FrameTransformMatrix
from ..VoD_lidar_radar.vod.frame import homogeneous_transformation

# import pypatchworkpp
import cv2

class VoDDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        radar_set = self.dataset_cfg.DATA_PATH.split('/')[-1]
        self.root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
                     'data', 'view_of_delft_PUBLIC', radar_set,)

        self.root_path_lidar = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
            'data', 'view_of_delft_PUBLIC', 'lidar')

        self.root_split_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
                     'data', 'view_of_delft_PUBLIC', radar_set, ('training' if self.split != 'test' else 'testing'))

        self.root_split_path_lidar = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
                     'data', 'view_of_delft_PUBLIC', 'lidar', ('training' if self.split != 'test' else 'testing'))

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()]

        self.VoD_infos = []
        self.include_VoD_data(self.mode)

        if self.dataset_cfg.get('DOWNSAMPLE',None):

            assert self.dataset_cfg.get('DOWNSAMPLE_RATE', None) is not None, "Downsample_Rate is not given"

            self.sample_rate = self.dataset_cfg.DOWNSAMPLE_RATE
            self.sample_method = self.dataset_cfg.DOWNSAMPLE_METHOD
            if self.sample_rate == 100:
                self.mask_file = None
            if self.sample_rate == 50:
                self.mask_file = os.path.join(self.root_path_lidar, self.sample_method, 'R_50')
            elif self.sample_rate == 25:
                self.mask_file = os.path.join(self.root_path_lidar, self.sample_method, 'R_25')
            elif self.sample_rate == 12.5:
                self.mask_file = os.path.join(self.root_path_lidar, self.sample_method, 'R_12')
            elif self.sample_rate == 6.25:
                self.mask_file = os.path.join(self.root_path_lidar, self.sample_method, 'R_06')
            elif self.sample_rate == 3.125:
                self.mask_file = os.path.join(self.root_path_lidar, self.sample_method, 'R_03')
            elif self.sample_rate == 1.5625:
                self.mask_file = os.path.join(self.root_path_lidar, self.sample_method, 'R_01')
            elif self.sample_rate == 0.78125:
                self.mask_file = os.path.join(self.root_path_lidar, self.sample_method, 'R_007')
            elif self.sample_rate == 0.390625:
                self.mask_file = os.path.join(self.root_path_lidar, self.sample_method, 'R_004')
            elif self.sample_rate == 0.0:
                self.mask_file = None
        
    def include_VoD_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading VoD dataset')
        VoD_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = os.path.join(self.root_path, info_path)
            if not os.path.isfile(info_path):
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                VoD_infos.extend(infos)

        self.VoD_infos.extend(VoD_infos)

        if self.logger is not None:
            self.logger.info('Total samples for VoD dataset: %d' % (len(VoD_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = os.path.join(self.root_path, ('training' if self.split != 'test' else 'testing'))

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()]

    def get_radar(self, idx):
        radar_file = os.path.join(self.root_split_path, 'velodyne', ('%s.bin' % idx))

        number_of_channels = 7  # ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
        points = np.fromfile(str(radar_file), dtype=np.float32).reshape(-1, number_of_channels)

        invalid_value = 0
        points_intensity = np.hstack([points, np.zeros([len(points), 1]) + invalid_value])

        return points_intensity

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.root_split_path_lidar, 'velodyne', ('%s.bin' % idx))
        number_of_channels = 4  # ['x', 'y', 'z', 'intensity']
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, number_of_channels)

        if self.dataset_cfg.DOWNSAMPLE:
            if not self.mask_file == None:
                with open(self.mask_file + '/' + idx + '.pkl', 'rb') as f:
                    pointmask = pickle.load(f)
                points = points[pointmask]

        # Transform Lidar Points to Radar coordinate system
        data_locations = KittiLocations(root_dir=str(self.root_path)[:-14], output_dir="")
        frame_data = FrameDataLoader(kitti_locations=data_locations,
                                     frame_number=idx)
        transforms = FrameTransformMatrix(frame_data)
        radar_xyz = np.hstack([points[:, :3], np.ones([len(points), 1])])
        points_transformed = homogeneous_transformation(radar_xyz, transforms.t_radar_lidar.round(3))
        
        points[:, :3] = points_transformed[:, :3]

        return points

    def get_lidar_radar(self, idx):
        lidar_file = os.path.join(self.root_split_path_lidar, 'velodyne', ('%s.bin' % idx))
        radar_file = os.path.join(self.root_split_path, 'velodyne', ('%s.bin' % idx))

        ### Load Lidar point cloud
        lidar_np = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        lidar_np_RCS = lidar_np[:, :4]

        # Transform Lidar Points to Radar coordinate system
        data_locations = KittiLocations(root_dir=str(self.root_path)[:-14], output_dir="")
        frame_data = FrameDataLoader(kitti_locations=data_locations,
                                     frame_number=idx)
        transforms = FrameTransformMatrix(frame_data)
        radar_xyz = np.hstack([lidar_np_RCS[:, :3], np.ones([len(lidar_np_RCS), 1])])
        points_transformed = homogeneous_transformation(radar_xyz, transforms.t_radar_lidar.round(3))

        lidar_np_RCS[:, :3] = points_transformed[:, :3]

        #### Load Radar point cloud ####
        radar_np = np.fromfile(str(radar_file), dtype=np.float32).reshape(-1, 7)
        radar_np_RCS = radar_np[:, :7]

        #### Join radar and lidar ####
        invalid_value = 0
        padding_lidar = np.hstack(
            [lidar_np_RCS[:, :3], np.zeros([len(lidar_np_RCS), 4]) + invalid_value, lidar_np_RCS[:, 3:]])

        if self.dataset_cfg.DOWNSAMPLE:
            if not self.mask_file == None:
                with open(self.mask_file + '/' + idx + '.pkl', 'rb') as f:
                    pointmask = pickle.load(f)
                padding_lidar = padding_lidar[pointmask]

        padding_radar = np.hstack([radar_np_RCS, np.zeros([len(radar_np_RCS), 1]) + invalid_value])
        lidar_radar_np = np.vstack([padding_radar, padding_lidar])

        if self.dataset_cfg.RETURN_DATASET == 'Lidar_Radar':
            return lidar_radar_np
        elif self.dataset_cfg.RETURN_DATASET == 'Radar':
            return padding_radar
        elif self.dataset_cfg.RETURN_DATASET == 'Lidar':
            return padding_lidar

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = os.path.join(self.root_split_path, 'image_2', ('%s.jpg' % idx))
        #assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = os.path.join(self.root_split_path, 'image_2', ('%s.jpg' % idx))
        #assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = os.path.join(self.root_split_path, 'label_2', ('%s.txt' % idx))
        #assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.root_split_path, 'calib', ('%s.txt' % idx))
        # calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        # plane_file = os.path.join(self.root_split_path, 'planes', ('%s.txt' % idx))
        # if not plane_file.exists():
        #     return None
        # return None
        # with open(plane_file, 'r') as f:
        #     lines = f.readlines()
        # lines = [float(i) for i in lines[3].split()]
        # plane = np.asarray(lines)
        #
        # # Ensure normal is always facing up, this is in the rectified camera coordinate
        # if plane[1] > 0:
        #     plane = -plane
        #
        # norm = np.linalg.norm(plane[0:3])
        # plane = plane / norm
        # return plane
        return None

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=1, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['id'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_radar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('VoD_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_radar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'id': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.VoD_infos[0].keys():
            return None, {}

        from .VoD_object_eval_python import eval as VoD_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.VoD_infos]
        ap_result_str, ap_dict = VoD_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def evaluation_vod(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.VoD_infos[0].keys():
            return None, {}

        from .VoD_object_eval_python import vod_official_evaluate as VoD_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.VoD_infos]
        ap_dict = VoD_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        ap_result_str = None

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.VoD_infos) * self.total_epochs

        return len(self.VoD_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.VoD_infos)

        info = copy.deepcopy(self.VoD_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            if hasattr(self.dataset_cfg, 'INSTANT_2D_VEL_EST') and self.dataset_cfg.INSTANT_2D_VEL_EST['SUPER_BBOX_LEARN']:
                gt_boxes_corners = box_utils.boxes_to_corners_3d(gt_boxes_lidar)

                # initialize array of past bounding boxes (shape: annos['id'] x 5 x 8 x 3 # [] x max number of consecutive frames x number of cobuoid corners x [])
                # array ordered according to annos['id'] at first axis
                #     7 -------- 4
                #    /|         /|
                #   6 -------- 5 .
                #   | |        | |
                #   . 3 -------- 0
                #   |/         |/
                #   2 -------- 1
                assoc_bbox_seq = -1000 * np.ones((annos['id'].size, 5, 8, 3))
                assoc_bbox_seq[:, 0, :, :] = gt_boxes_corners
                # assoc_bbox_seq[:, 0, 0, 0] = annos['id']
                assoc_head_seq = -10 * np.ones((annos['id'].size, 5))
                assoc_head_seq[:, 0] = gt_boxes_lidar[:, -1]
                assoc_exist_seq = np.full((annos['id'].size, 5), False)
                assoc_exist_seq[:, 0] = True

                for i in range(4):
                    # assuming every entry from 0 to N-1 exists
                    if index - (i+1) >= 0:
                        info_ = copy.deepcopy(self.VoD_infos[index - (i+1)])
                        annos_ = info_['annos']
                        annos_ = common_utils.drop_info_with_name(annos_, name='DontCare')

                        assoc_exist_seq[:, (i + 1)] = np.isin(annos['id'], annos_['id'], assume_unique=True)

                        annos_filtered_ids = np.isin(annos_['id'], annos['id'], assume_unique=True)
                        annos_filtered_dict = {id: idx for idx, id in enumerate(annos['id'])}
                        rel_id_order = np.array([annos_filtered_dict[id] for id in annos_['id'][annos_filtered_ids]])
                        if rel_id_order.size == 0:
                            continue

                        loc_, dims_, rots_ = annos_['location'][annos_filtered_ids, :], annos_['dimensions'][annos_filtered_ids, :], annos_['rotation_y'][annos_filtered_ids]
                        gt_boxes_camera_ = np.concatenate([loc_, dims_, rots_[..., np.newaxis]], axis=1).astype(np.float32)
                        gt_boxes_lidar_ = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera_, calib)
                        gt_boxes_corners_ = box_utils.boxes_to_corners_3d(gt_boxes_lidar_)
                        assoc_bbox_seq[rel_id_order, (i+1), :, :] = gt_boxes_corners_
                        assoc_head_seq[rel_id_order, (i+1)] = gt_boxes_lidar_[:, -1]


                        gt_boxes_lidar_reordered = -1000 * np.ones_like(gt_boxes_lidar)
                        gt_boxes_lidar_reordered[rel_id_order, :] = gt_boxes_lidar_
                        input_dict['gt_boxes_-' + str(int(i+1))] = gt_boxes_lidar_reordered
                    else:
                        continue

                # determine min and max of values for corners along all three dimensions of time series of 3D bounding box per object
                #     7 -------- 4
                #    /|         /|
                #   6 -------- 5 .
                #   | |        | |
                #   . 3 -------- 0
                #   |/         |/
                #   2 -------- 1
                # fill with object-wise minimum to prevent from accounting for default values: -1
                default_mask = (assoc_head_seq == -10)
                assoc_head_seq_ = np.ma.masked_array(assoc_head_seq, mask=default_mask) # https://numpy.org/doc/stable/reference/maskedarray.generic.html
                assoc_head_seq_mean = assoc_head_seq_.mean(axis=1)
                default_mask = (assoc_bbox_seq == -1000)
                assoc_bbox_seq_ = np.ma.masked_array(assoc_bbox_seq, mask=default_mask) # https://numpy.org/doc/stable/reference/maskedarray.generic.html
                temp_diag_corner_hull = np.stack((np.amax(assoc_bbox_seq_[:,:,:,0], axis=(1,2)), np.amin(assoc_bbox_seq_[:,:,:,0], axis=(1,2)), # x_max, x_min
                                                  np.amax(assoc_bbox_seq_[:,:,:,1], axis=(1,2)), np.amin(assoc_bbox_seq_[:,:,:,1], axis=(1,2)), # y_max, y_min
                                                  np.amax(assoc_bbox_seq_[:,:,:,2], axis=(1,2)), np.amin(assoc_bbox_seq_[:,:,:,2], axis=(1,2)) # z_max, z_min
                                                   ), axis=1)
                temp_diag_corner_hull = np.ma.getdata(temp_diag_corner_hull)
                gt_boxes_lidar__ = np.stack(((temp_diag_corner_hull[:,0]+temp_diag_corner_hull[:,1])/2, # x_c
                                             (temp_diag_corner_hull[:, 2] + temp_diag_corner_hull[:, 3]) / 2, # y_c
                                             (temp_diag_corner_hull[:, 4] + temp_diag_corner_hull[:, 5]) / 2, # z_y
                                             np.abs(temp_diag_corner_hull[:, 0] - temp_diag_corner_hull[:, 1]), # l
                                             np.abs(temp_diag_corner_hull[:, 0] - temp_diag_corner_hull[:, 1]), # w
                                             np.abs(temp_diag_corner_hull[:, 0] - temp_diag_corner_hull[:, 1]), # h
                                             np.ma.getdata(assoc_head_seq_mean) # mean heading angle
                                             ), axis=1)
                input_dict['gt_super-box'] = gt_boxes_lidar__

                input_dict['gt_obj_exist'] = assoc_exist_seq

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_radar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

            # Load teacher dataset
            if hasattr(self.dataset_cfg, 'TEACHERSET'):
                if self.dataset_cfg.TEACHERSET == 'Lidar':
                    points = self.get_lidar(sample_idx)
                    if self.dataset_cfg.FOV_POINTS_ONLY:
                        pts_rect = calib.lidar_to_rect(points[:, 0:3])
                        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                        points = points[fov_flag]
                    input_dict['points_tea'] = points
                elif self.dataset_cfg.TEACHERSET == 'LidarRadar':
                    points = self.get_lidar_radar(sample_idx)
                    if self.dataset_cfg.FOV_POINTS_ONLY:
                        pts_rect = calib.lidar_to_rect(points[:, 0:3])
                        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                        points = points[fov_flag]
                    input_dict['points_tea'] = points
                else:
                    raise NotImplementedError

        if "camera_opening_angle" in get_item_list:
            input_dict['fovx'], input_dict['fovy'], _, _, _ = cv2.calibrationMatrixValues(calib.P2[:3,:3], np.flip(img_shape), 1, 1)

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = VoD_utils.calib_to_matricies(calib)

        data_dict = self.prepare_data(data_dict=input_dict)

        if np.sum(np.isnan(data_dict['points'])) > 0:
            data_dict['points'][np.isnan(data_dict['points'])] = 0.001

        data_dict['image_shape'] = img_shape
        return data_dict


def create_VoD_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = VoDDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('VoD_infos_%s.pkl' % train_split)
    val_filename = save_path / ('VoD_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'VoD_infos_trainval.pkl'
    test_filename = save_path / 'VoD_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    VoD_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(VoD_infos_train, f)
    print('VoD info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    VoD_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(VoD_infos_val, f)
    print('VoD info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(VoD_infos_train + VoD_infos_val, f)
    print('VoD info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    VoD_infos_test = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(VoD_infos_test, f)
    print('VoD info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_VoD_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_VoD_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'view_of_delft_PUBLIC' / dataset_cfg.POINT_CLOUD_TYPE[0],
            save_path=ROOT_DIR / 'data' / 'view_of_delft_PUBLIC' / dataset_cfg.POINT_CLOUD_TYPE[0]
        )
