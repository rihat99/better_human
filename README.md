# better_human
Collection of articulated human models


## Convert SMPL family models

SMPL famaly models sometimes use Chumpy library and thus requires old version of numpy. To use them together with modern libraries, we first need to covert them. First download original models you need:

1) [SMPL](https://smpl.is.tue.mpg.de/) (work in progress)
2) [SMPL-H](https://mano.is.tue.mpg.de/) (not yet supported)
3) [SMPL-X](https://smpl-x.is.tue.mpg.de/) (not yet supported)

Create temporary conda environment that will use old version of numpy
```
conda create -n smpl_convert python=3.10
conda activate smpl_convert
pip install numpy==1.23.5 chumpy
```

Convert SMPL from .pkl (contain chumpy arrays) to .npz
```
python convert.py --model SMPL --file PATH_TO_SMPL/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl --save SAVE_PATH/SMPL_NEUTRAL
```

## Installation

```
conda create -n better_human python=3.10
```