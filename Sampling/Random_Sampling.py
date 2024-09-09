'''
Save indices for random sampling of the Lidar Point cloud
'''

import os
import numpy as np
import pickle

# Get input data
DATA_PATH = '../data/view_of_delft_PUBLIC/lidar/training/velodyne'
DATA_PATH_OUT = '../data/view_of_delft_PUBLIC/lidar'
data_filelist = os.listdir(DATA_PATH)
data_filelist.sort()

def random_sampling(file_str):
    print(file_str)
    # Load the LIDAR data
    lidar_file = os.path.join(DATA_PATH, file_str)
    lidar_np = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    lidar_np_xyz = lidar_np[:, :4]

    # Initialize variables
    num_ofpoints = lidar_np_xyz.shape[0]
    list_ofidcs = np.arange(num_ofpoints)
    sample_indices = list_ofidcs
    reduction_factors = [50, 25, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625]
    folder_paths = [
        "Sampling_Random/R_50/",
        "Sampling_Random/R_25/",
        "Sampling_Random/R_12/",
        "Sampling_Random/R_06/",
        "Sampling_Random/R_03/",
        "Sampling_Random/R_01/",
        "Sampling_Random/R_007/",
        "Sampling_Random/R_004/"
    ]

    # Loop through the reduction factors and sample the points
    for factor, folder_path in zip(reduction_factors, folder_paths):
        sample_indices = np.random.choice(sample_indices, size=int(len(sample_indices) * 0.5), replace=False)

        # Create directory if it doesn't exist
        os.makedirs(os.path.join(DATA_PATH_OUT, folder_path), exist_ok=True)

        # Save the sampled indices
        with open(os.path.join(DATA_PATH_OUT, folder_path, file_str[:-4] + '.pkl'), "wb") as fp:
            pickle.dump(sample_indices, fp)

# Parallel execution of every file
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(random_sampling, data_filelist)