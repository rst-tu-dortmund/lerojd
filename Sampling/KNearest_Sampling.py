'''
Save indices for knearest neighbor sampling of the Lidar Point cloud
'''

import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

DATA_PATH = '../data/view_of_delft_PUBLIC/lidar/training/velodyne'
DATA_PATH_Radar = '../data/view_of_delft_PUBLIC/radar/training/velodyne'
DATA_PATH_OUT = '../data/view_of_delft_PUBLIC/lidar'

data_filelist = os.listdir(DATA_PATH)
data_filelist.sort()

def knearest_sampling(file_str):
    # Load lidar data
    print(file_str)
    lidar_file = os.path.join(DATA_PATH, file_str)
    lidar_np = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    lidar_np_xyz = lidar_np[:, :3]

    # Load radar data and adjust the lidar coordinate frame
    radar_file = os.path.join(DATA_PATH_Radar, file_str)
    radar_np = np.fromfile(str(radar_file), dtype=np.float32).reshape(-1, 7)
    radar_np_xyz = radar_np[:, :3]
    radar_np_xyz[:, 0] = radar_np_xyz[:, 0] + 2.5065
    radar_np_xyz[:, 1] = radar_np_xyz[:, 1] + 0.2025
    radar_np_xyz[:, 2] = radar_np_xyz[:, 2] - 0.9356

    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(radar_np_xyz)
    min_y_to_x = x_nn.kneighbors(lidar_np_xyz)[0]
    chamfer_sort = np.argsort(min_y_to_x, axis=0)
    num_ofpoints = lidar_np_xyz.shape[0]

    # Reduction factors and corresponding folder paths
    reduction_factors = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
    folder_paths = [
        "Sampling_KN/R_50/",
        "Sampling_KN/R_25/",
        "Sampling_KN/R_12/",
        "Sampling_KN/R_06/",
        "Sampling_KN/R_03/",
        "Sampling_KN/R_01/",
        "Sampling_KN/R_007/",
        "Sampling_KN/R_004/"
    ]

    # Loop through the reduction factors and save the corresponding indices
    for factor, folder_path in zip(reduction_factors, folder_paths):
        sample_indices = chamfer_sort[:int(num_ofpoints * factor), 0]

        # Create directory if it doesn't exist
        os.makedirs(os.path.join(DATA_PATH_OUT, folder_path), exist_ok=True)

        # Save the sampled indices
        with open(os.path.join(DATA_PATH_OUT, folder_path, file_str[:-4] + '.pkl'), "wb") as fp:
            pickle.dump(sample_indices, fp)

# Parallel execution of every file
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(knearest_sampling, data_filelist)