# PlaneNet

###### *PyTorch implementation of TCSVT paper: "Surface-continuous Scene Representation for Light Field Depth Estimation via Planarity Prior"*.

[Paper](https://ieeexplore.ieee.org/abstract/document/10810496)
#### Requirements

- python 3.6
- pytorch 1.8.0
- ubuntu 18.04

### Installation

First you have to make sure that you have all dependencies in place. 

You can create an anaconda environment called PlaneNet using

```
conda env create -f PlaneNet.yaml
conda activate PlaneNet
```

##### Dataset: 

Light Field Dataset: We use [HCI 4D Light Field Dataset](https://lightfield-analysis.uni-konstanz.de/) for training and test. Please first download light field dataset with its full-view depth information, and put them into corresponding folders in ***data/HCInew***.


## PlaneNet
Coming Soon!


## PlaneDistgDisp

##### Model weights: 
Please download the model weights from [Google Drive](https://drive.google.com/drive/folders/187RP68_Q1UOVkzXKAXdZ75i4klWp20Jx?usp=sharing), and put the "model_best.pt" in the ***out/PlaneDistgDisp/HCInew***.

##### To train, run:

```
python train.py --config configs/HCInew/PlaneDistgDisp.yaml 
```

##### To generate, run:

```
python generate.py --config configs/pretrained/HCInew/PlaneDistgDisp_pretrained.yaml 
```



**If you find our code or paper useful, please consider citing:**
```
@article{chen2024surface,
  title={Surface-continuous Scene Representation for Light Field Depth Estimation via Planarity Prior},
  author={Chen, Rongshan and Sheng, Hao and Yang, Da and Cui, Zhenglong and Cong, Ruixuan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```
