'''
This script is used to remove the duplicate points in the View of Delft lidar point cloud.
'''

import os

import numpy as np

DATA_PATH = '../data/view_of_delft_PUBLIC/lidar/training/velodyne'
data_fillist = os.listdir(DATA_PATH)
data_fillist.sort()

os.makedirs('../data/view_of_delft_PUBLIC/lidar/training/velodyne_nondouble', exist_ok=True)

for file_str in data_fillist:
    lidar_file = os.path.join(DATA_PATH, file_str)
    lidar_np = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    lidar_np = lidar_np[:, :4]

    # Limit lidar to radar view range [0<x<52.2, -25.6<y<25.6, z]
    lidar_np = lidar_np[lidar_np[:,0]>0]
    lidar_np = lidar_np[lidar_np[:,0]<52.2]
    lidar_np = lidar_np[lidar_np[:,1]>-25.6]
    lidar_np = lidar_np[lidar_np[:,1]<25.6]

    unique_array = np.unique(lidar_np, axis=0)

    unique_array.tofile('../data/view_of_delft_PUBLIC/lidar/training/velodyne_nondouble/' + file_str)

    print(file_str)
