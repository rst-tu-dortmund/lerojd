# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 


## Dataset Preparation

### Step 1: Prepare the datasets for OpenPCDet
Please follow the OpenPCDet [tutorial](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to 
prepare needed datasets.

### Step 2: Compute the subsampling of the lidar point cloud
For the subsampling of the lidar point cloud the indices of points to remove from the pointcloud have to be computed.
This can be done by running the following command:
* The View of Delft lidar point cloud contains every point two times [(Corresponding Git Issue on VoD repository)](https://github.com/tudelft-iv/view-of-delft-dataset/issues/72). This does not affect the lidar training substantially, but they need to be removed for the sampling of points. Additionally we cap the lidar point cloud to the radar view range.
```shell
python Sampling/RemoveDoublePoints.py
```
* The filtered point cloud is saved in the folder "velodyne_nondouble". The original folder "velodyne" has to be removed and replaced by the folder "velodyne_nondouble" renamed to "velodyne".
* The sampling indices of the point cloud can then be computed with one of the sampling strategies (Random sampling as an example):
```shell
python Sampling/Random_Sampling.py
```

## Training utilizing Knowledge Distillation
[//]: # ( TODO)
### Step 1: Train a teacher model (PointPillars as example)
```shell
python train.py --cfg_file cfgs/VoD_teacher/pp_lidarradar_voxel25.yaml
```

### Step 2: Train a Student Model (PointPillars as example)
Modify following keys in the student distillation config
```yaml
# cfgs/waymo_models/cp-pillar/cp-pillar-v0.4_sparsekd.yaml
TEACHER_CKPT: ${PATH_TO_TEACHER_CKPT}
PRETRAINED_MODEL: ${PATH_TO_TEACHER_CKPT}
```
Run the training config
```shell
python train.py --cfg_file cfgs/VoD_student/PointPillars.yaml
``` 

## Training utilizing Multi Stage Training Method (MSTM)
[//]: # ( TODO)
### Step 1: Train a teacher model (PointPillars as example)
```shell
python train_MSTM.py --folder VoD_MSTM --model pp_lidar_random
```

