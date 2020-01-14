### A-SCN: *Attentional ShapeContextNet for Point Cloud Recognition*
Created by <a href="http://vcl.ucsd.edu/~sxie/" target="_blank">Saining Xie*</a>, <a href="">Sainan Liu*</a>, <a href="" target="_blank">Zeyu Chen</a>, <a href="https://pages.ucsd.edu/~ztu/" target="_blank">Zhuowen Tu</a> from University of California, San Diego.

<img src="https://github.com/umyta/A-SCN/blob/master/doc/teaser.png" width="40%">


### Introduction
This repository provides a sample code for the paper [Attentional ShapeContextNet for Point Cloud Recognition](http://pages.ucsd.edu/~ztu/publication/cvpr18_ascn.pdf).
 In the paper, we introduce a neural network based algorithm by adopting the concept of shape context kernel
 for 3D shape recognition. The resulting network is ShapeContextNet (SCN), which has 
 hierarchical modules that can represent the intrinsic property of object points by
 capturing and propagating both the local part and the global shape information. Additionally, we propose
 Attentional ShapeContextNet (A-SCN) which automate the process for the contextual region selection,
 feature aggregation, and feature transformation. 
 
 In this repository, we provide a sample code for A-SCN. 

### Installation
#### Tensorflow
We use the same set of datasets from PointNet, and we have run our code in the following environment:

- python 3.6
- tensorflow 1.11.0
- CUDA 9.0
- cuDNN 7
- Ubuntu 16.04

To install h5py for Python:
```
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

To run this code, we use a docker image that is built on top of `nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04`,
similar docker files can be found from this third-party [repository](https://github.com/ufoym/deepo).

For shape classification, part segmentation and semantic segmentation, please follow the instructions under the [classification](https://github.com/umyta/A-SCN/blob/master/classification), [part_seg](https://github.com/umyta/A-SCN/blob/master/part_seg) and [sem_seg](https://github.com/umyta/A-SCN/blob/master/sem_seg) folders respectively.

### Acknowledgement
Part of this code is built on top of [PointNet](https://github.com/charlesq34/pointnet) / [PointNet++](https://github.com/charlesq34/pointnet2) .
 
### License
Our code is released under MIT License (see LICENSE file for details).

### Citation
If you find our work useful in your research, please consider citing:

        @article{saining2018ascn,
          title={Attentional ShapeContextNet for Point Cloud Recognition},
          author={Xie, Saining and Liu, Sainan and Chen, Zeyu and Tu, Zhuowen},
          year={2018}
        }
