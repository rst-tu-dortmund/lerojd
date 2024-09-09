import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
from pcdet.models.backbones_3d.vfe import vfe_template


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        # relative x y z compared to the voxel center
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict


class Radar7PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        num_point_features = 0
        self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
        self.use_xyz = self.model_cfg.USE_XYZ  # USE_XYZ
        self.with_distance = self.model_cfg.USE_DISTANCE  # USE_DISTANCE
        self.selected_indexes = []

        ## check if config has the correct params, if not, throw exception
        radar_config_params = ["USE_RCS", "USE_VR", "USE_VR_COMP", "USE_TIME", "USE_ELEVATION", "USE_PCA"]

        if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
            self.use_RCS = self.model_cfg.USE_RCS
            self.use_vr = self.model_cfg.USE_VR
            self.use_vr_comp = self.model_cfg.USE_VR_COMP
            self.use_time = self.model_cfg.USE_TIME
            self.use_elevation = self.model_cfg.USE_ELEVATION
            self.use_pca = self.model_cfg.USE_PCA

        else:
            raise Exception("config does not have the right parameters, please use a radar config")

        self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time', 'a1', 'a2', 'b1', 'b2', 'c1', 'c2']

        num_point_features += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.rcs_ind = self.available_features.index('rcs')
        self.vr_ind = self.available_features.index('v_r')
        self.vr_comp_ind = self.available_features.index('v_r_comp')
        self.time_ind = self.available_features.index('time')

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_RCS:  # add 1 if RCS is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.vr_comp_ind)

        if self.use_time:  # add 1 if time is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes

        if self.use_pca:
            self.a1 = self.available_features.index('a1')
            self.a2 = self.available_features.index('a2')
            self.b1 = self.available_features.index('b1')
            self.b2 = self.available_features.index('b2')
            self.c1 = self.available_features.index('c1')
            self.c2 = self.available_features.index('c2')

            num_point_features += 6  # x, y, z
            self.selected_indexes.extend((self.a1, self.a2, self.b1, self.b2, self.c1, self.c2))

        ### LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features))
        print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")

        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        ## saving size of the voxel
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]  # number of outputs in last output channel

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        ## coordinate system notes
        # x is pointing forward, y is left right, z is up down
        # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']

        if not self.use_elevation:  # if we ignore elevation (z) and v_z
            voxel_features[:, :, self.z_ind] = 0  # set z to zero before doing anything

        orig_xyz = voxel_features[:, :, :self.z_ind + 1]  # selecting x y z

        # calculate mean of points in pillars for x y z and save the offset from the mean
        # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
        points_mean = orig_xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = orig_xyz - points_mean  # offset from cluster mean

        # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
        f_center = torch.zeros_like(orig_xyz)
        f_center[:, :, 0] = voxel_features[:, :, self.x_ind] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, self.y_ind] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, self.z_ind] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        voxel_features = voxel_features[:, :, self.selected_indexes]  # filtering for used features

        features = [voxel_features, f_cluster, f_center]

        if self.with_distance:  # if with_distance is true, include range to the points as well
            points_dist = torch.norm(orig_xyz, 2, 2, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
            features.append(points_dist)

        ## finishing up the feature extraction with correct shape and masking
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict


class LidarRadar7PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        num_point_features = 0
        self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
        self.use_xyz = self.model_cfg.USE_XYZ
        self.with_distance = self.model_cfg.USE_DISTANCE
        self.selected_indexes = []

        ## check if config has the correct params, if not, throw exception
        radar_config_params = ["USE_RCS", "USE_VR", "USE_VR_COMP", "USE_TIME", "USE_ELEVATION", "USE_REFLECTANCE"]

        if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
            self.use_RCS = self.model_cfg.USE_RCS
            self.use_vr = self.model_cfg.USE_VR
            self.use_vr_comp = self.model_cfg.USE_VR_COMP
            self.use_time = self.model_cfg.USE_TIME
            self.use_elevation = self.model_cfg.USE_ELEVATION
            self.use_reflectance = self.model_cfg.USE_REFLECTANCE
        else:
            raise Exception("config does not have the right parameters, please use a radar config")

        self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time', 'reflectance']

        num_point_features += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.rcs_ind = self.available_features.index('rcs')
        self.vr_ind = self.available_features.index('v_r')
        self.vr_comp_ind = self.available_features.index('v_r_comp')
        self.time_ind = self.available_features.index('time')
        self.reflectance_ind = self.available_features.index('reflectance')

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_RCS:  # add 1 if RCS is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.vr_comp_ind)

        if self.use_time:  # add 1 if time is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes

        if self.use_reflectance:
            num_point_features += 1
            self.selected_indexes.append(self.reflectance_ind)

        ### LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features))
        print("6 of these are 2 * (x y z)  coordinates relative to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")

        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        ## saving size of the voxel
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]  # number of outputs in last output channel

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        ## coordinate system notes
        # x is pointing forward, y is left right, z is up down
        # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']

        if not self.use_elevation:  # if we ignore elevation (z) and v_z
            voxel_features[:, :, self.z_ind] = 0  # set z to zero before doing anything

        orig_xyz = voxel_features[:, :, :self.z_ind + 1]  # selecting x y z

        # calculate mean of points in pillars for x y z and save the offset from the mean
        # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
        points_mean = orig_xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = orig_xyz - points_mean  # offset from cluster mean

        # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
        f_center = torch.zeros_like(orig_xyz)
        f_center[:, :, 0] = voxel_features[:, :, self.x_ind] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, self.y_ind] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, self.z_ind] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        voxel_features = voxel_features[:, :, self.selected_indexes]  # filtering for used features

        features = [voxel_features, f_cluster, f_center]

        if self.with_distance:  # if with_distance is true, include range to the points as well
            points_dist = torch.norm(orig_xyz, 2, 2, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
            features.append(points_dist)

        ## finishing up the feature extraction with correct shape and masking
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict