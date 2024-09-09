# Installation

### Requirements
All the code is tested in the following environment:
* Ubuntu 20.04
* Python 3.9
* PyTorch 1.13
* CUDA 11.6 
* spconv 2.3

### Install `pcdet v0.6.0`
NOTE: Please re-install `pcdet v0.6.0` by running `python setup.py develop` even if you have already installed previous versions.

a. Clone this repository.
```shell
git clone https://github.com/rst-tu-dortmund/lerojd.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries:

```
pip install -r requirements.txt
```

* Install the latest SparseConv library using pip ,see the official documents of [spconv](https://github.com/traveller59/spconv).
  
c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```
