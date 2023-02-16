## Pytorch ver.

<img src=./assets/compare.png width=600 height=600>

#### Requirements

PyTorch1.1+, Python3.7+, CUDA 10.2+

#### Kitti data

Our training data is the same with other self-supervised monocular depth estimation methods, please refer to [featdepth](https://github.com/sconlyshootery/FeatDepth) or [monodepth2](https://github.com/nianticlabs/monodepth2) to prepare the training data.

#### Training

You can use following command to train the model:

```
python train.py --config /path/to/cfg_kitti_fm.py --work_dir ./output/ --gpus 0,1,2,3
```

#### Evaluation

You can use following command to evaluate

```
python scripts/eval_depth_pp.py
```

You also can set `save_error` and `zoom_post_process` to save disp images or use post-processing.

#### Pretrained weights

You can download our pretrained weight [here](https://drive.google.com/file/d/1g0xALvd4hEKbW3dnnqC8kfQYxCmJgHvy/view?usp=sharing)

#### Configurations

More detailed settings are in the config file (max epoch, batchsize, height, weight, pretrained model path, network...)

#### Acknowledgement

Thanks the authors for their works: 

[Monodepth2](https://github.com/nianticlabs/monodepth2)

[Featdepth](https://github.com/sconlyshootery/FeatDepth)

[DIFFNet](https://github.com/brandleyzhou/diffnet)
