## Paddle ver.

#### Requirements

PyTorch1.10+, Python3.6+, Paddle2.2.1+, cuda 10.1+

#### Kitti data

Our training data is the same with other self-supervised monocular depth estimation methods, please refer to [featdepth](https://github.com/sconlyshootery/FeatDepth) or [monodepth2](https://github.com/nianticlabs/monodepth2) to prepare the training data.

#### KITTI inference

**Noted:** pretrained weights

You can use following command to evaluate:

```
python test_kitti_padd.py
```

You can predict scaled disparity for a single image with:

```
python infer_simple_padd.py
```

