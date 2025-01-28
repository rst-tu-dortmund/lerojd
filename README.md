# LEROjD: Lidar Extended Radar-Only Object Detection](htt

---

This repository contains the official implementation of the ECCV 2024 paper [LEROjD: Lidar Extended Radar-Only Object Detection](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07776.pdf),


## Changelog
[2024-08-] Initial release\
[2025-01-] Updated README, added pretrained models

## Introduction
Our code is based on [SparseKD](https://github.com/CVMI-Lab/SparseKD) which itself is based on [OpenPCDet v0.5.2](https://github.com/open-mmlab/OpenPCDet/tree/v0.5.2).
Some changes of [OpenPCDet v0.6.0](https://github.com/open-mmlab/OpenPCDet/tree/master), like DSVT, are integrated in this codebase.

We only supply code for execution on the [View of Delft]((https://github.com/tudelft-iv/view-of-delft-dataset)). The code should be applicable to other datasets with minor changes.

Due to the limited availability of evaluation on the [View of Delft](https://github.com/tudelft-iv/view-of-delft-dataset) Test dataset we use a different split of the dataset for training and evaluation.
This modified split is available in the [data](data/view_of_delft_PUBLIC/lidar/ImageSets) folder.

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage of OpenPCDet.

## Pretrained Models
| Model (PointPillars)                                                              | download                                                          |
|-----------------------------------------------------------------------------------|-------------------------------------------------------------------|
| $\text{RL}^{\text{MSTM}}_{\text{1-1/16 / vox} }\rightarrow\text{R} $              | [model](https://tu-dortmund.sciebo.de/s/G2CwVU6QRa6CyTs/download) |
| $\text{RL}^{\text{SSTM}}_{\text{1/4 }} \rightarrow\text{R}^{\text{label}}$        | [model](https://tu-dortmund.sciebo.de/s/M9wn9qh67xJF93U/download)          |
| $\text{RL}^{\text{SSTM}}_{\text{1/4 / knn}} \rightarrow\text{R}^{\text{feature}}$ | [model](https://tu-dortmund.sciebo.de/s/itgXf4DrSS8NUJM/download)          |



## License
This codebase is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
Our code is heavily based on [SparseKD](https://github.com/CVMI-Lab/SparseKD) and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 
Thanks to the authors of SparseKD as well as the OpenPCDet Development Team for their awesome codebase.

