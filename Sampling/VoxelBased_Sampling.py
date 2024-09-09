'''
Save indices for voxel based sampling of the Lidar Point cloud
'''

import os
import numpy as np
import pickle
from pcdet.datasets.processor.point_cloud_ops import points_to_voxel

DATA_PATH = '../data/view_of_delft_PUBLIC/lidar/training/velodyne'
DATA_PATH_OUT = '../data/view_of_delft_PUBLIC/lidar'

data_filelist = os.listdir(DATA_PATH)
data_filelist.sort()

def voxel_sampling(file_str):
    print(file_str)
    lidar_file = os.path.join(DATA_PATH, file_str)
    lidar_np = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    lidar_np = lidar_np[:, :4]

    # Voxelize Lidar Point Cloud
    points = lidar_np[:, :3]
    voxel_size = [1.0, 1.0, 1.0]
    coors_range = [0, -25.6, -10, 52.2, 25.6, 10]
    max_points = 50000
    max_voxels = 20000

    voxels, coordinate, num_ofpoints_per_voxel = points_to_voxel(points, voxel_size, coors_range, max_points, False, max_voxels)

    # Output folder for iteration
    folder_paths = [
        "Sampling_VoxelBased/R_50/",
        "Sampling_VoxelBased/R_25/",
        "Sampling_VoxelBased/R_12/",
        "Sampling_VoxelBased/R_06/",
        "Sampling_VoxelBased/R_03/",
        "Sampling_VoxelBased/R_01/",
        "Sampling_VoxelBased/R_007/",
        "Sampling_VoxelBased/R_004/"
    ]

    # First definition of maximum threshold and total number of points
    max_threshold = num_ofpoints_per_voxel.max()
    num_ofpoints = num_ofpoints_per_voxel.sum()

    # Iterate over thinning stages
    for folder_path in folder_paths:
        # Calculate minimum number of points per voxel, so that at least 75% of the points are in voxels with more than the minimum number of points
        for threshold in range(max_threshold, -1, -1):
            points_in_voxel_above_threshold = num_ofpoints_per_voxel[num_ofpoints_per_voxel > threshold].sum()
            percentage_above_treshold = points_in_voxel_above_threshold / num_ofpoints
            Count_PV = np.clip(num_ofpoints_per_voxel - threshold, a_min=0, a_max=max_points).sum()
            if percentage_above_treshold > 0.75 and Count_PV > 0.5*num_ofpoints:
                min_point_per_voxel = threshold
                max_threshold = threshold
                break
            if threshold == 0:
                min_point_per_voxel = 0
                max_threshold = 0

        percentage_above_halftreshold = (0.5*num_ofpoints)/Count_PV # Remove this part from the voxels with too many points

        # Remove Points from voxels if number of points in a voxel is above the threshold
        for vox_id in range(voxels.shape[0]):
            num_points_invoxel = num_ofpoints_per_voxel(vox_id)
            if num_points_invoxel > min_point_per_voxel:
                if min_point_per_voxel == 0 and num_points_invoxel == 1:
                    if np.random.rand() > 0.5:
                        voxels[vox_id, 0, :] = 0.0
                        num_ofpoints_per_voxel[vox_id] = points_to_keep + min_point_per_voxel
                else:
                    points_to_keep = int(num_points_invoxel * (1-percentage_above_halftreshold))
                    voxels[vox_id, points_to_keep+min_point_per_voxel:, :] = 0.0
                    num_ofpoints_per_voxel[vox_id] = points_to_keep+min_point_per_voxel

        # Get indices of the remaining points
        List_of_points_to_keep = []
        for vox_id in range(voxels.shape[0]):
            for point_id in range(voxels.shape[1]):
                if np.sum(voxels[vox_id, point_id, :]) != 0.0:
                    id_x = np.where(points[:, 0] == voxels[vox_id, point_id, 0])[0]
                    id_xy = id_x[points[id_x, 1] == voxels[vox_id, point_id, 1]]
                    id_xyz = id_xy[points[id_xy, 2] == voxels[vox_id, point_id, 2]]
                    List_of_points_to_keep.append(id_xyz[0])
                else:
                    break
        sample_indices = np.array(List_of_points_to_keep)

        # Create directory if it doesn't exist
        os.makedirs(os.path.join(DATA_PATH_OUT, folder_path), exist_ok=True)

        # Save the sampled indices
        with open(os.path.join(DATA_PATH_OUT, folder_path, file_str[:-4] + '.pkl'), "wb") as fp:
            pickle.dump(sample_indices, fp)

        # Get new count of total points
        num_ofpoints = np.count_nonzero(voxels) / 3

# Parallel execution of every file
import concurrent.futures
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    executor.map(voxel_sampling, data_filelist)