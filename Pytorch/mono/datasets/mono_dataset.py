from __future__ import absolute_import, division, print_function
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
        
class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 is_train=False,
                 img_ext='.jpg',
                 gt_depth_path=None,
                 CZDA=False,
                 test=False
                 ):
        super(MonoDataset, self).__init__()
        self.interp = Image.ANTIALIAS
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.gt_depth_path = gt_depth_path
        self.CZDA = CZDA
        self.test = test
        self.to_tensor = transforms.ToTensor()

        # Need to specify augmentations differently in pytorch 1.0 compared with 0.4
        if int(torch.__version__.split('.')[0]) > 0:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        else:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)

        self.flag = np.zeros(self.__len__(), dtype=np.int64)

        if not is_train and self.gt_depth_path is not None:
            self.gt_depths = np.load(gt_depth_path,
                                     allow_pickle=True,
                                     fix_imports=True, encoding='latin1')["data"]

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                inputs[(n, im, 0)] = self.resize(inputs[(n, im, - 1)])
        
        for k in list(inputs):
            if "color" in k:
                f = inputs[k]
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if i == 0:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def zoom_crop(self, inputs, scale=0.5):
        for k in list(inputs):
            if "color_aug" in k:
                n, im, i = k
                if scale < 1.0:
                    inputs[("color_aug_zoom", im, 0)] = self.center_crop(inputs[(n, im, 0)], scale)
                else:
                    inputs[("color_aug_zoom", im, 0)] = inputs[(n, im, 0)]
            if "color" in k:
                n, im, i = k
                if scale < 1.0:
                    inputs[("color", im, i)] = self.center_crop(inputs[(n, im, i)], scale)
                else:
                    inputs[("color", im, i)] = inputs[(n, im, i)]

    def zoom_crop_test(self, inputs, scale=0.5, scale_twc=0):
        for k in list(inputs):
            if "color_aug" in k:
                n, im, i = k
                inputs[("color_zoom_pp", im, 0)] = self.center_crop(inputs[(n, im, 0)], scale)
                if scale_twc > 0:
                    inputs[("color_zoom_pp_twice", im, 0)] = self.center_crop(inputs[(n, im, 0)], scale_twc)
                    
    def zoom_crop_xy(self, inputs, scale=1.0, scale_=1.0):
        for k in list(inputs):
            if "color_aug" in k:
                n, im, i = k
                if scale < 1.0:
                    inputs[("color_aug_zoom", im, 0)] = self.center_crop(inputs[(n, im, 0)], scale, scale_)
                else:
                    inputs[("color_aug_zoom", im, 0)] = inputs[(n, im, 0)]
            if "color" in k:
                n, im, i = k
                if scale < 1.0:
                    inputs[("color", im, i)] = self.center_crop(inputs[(n, im, i)], scale, scale_)
                else:
                    inputs[("color", im, i)] = inputs[(n, im, i)]
    
    def center_crop(self, img, scale=0.5, scale_=0.1):
        _, h, w = img.size()
        if scale_ == 0.1:
            h_start = int((h - h*scale) / 2)
            h_end = int(h - h_start)
        else:
            h_start = int((h - h*scale_) / 2)
            h_end = int(h - h_start)
        w_start = int((w - w * scale) / 2)
        w_end = int(w - w_start)

        img_crop = img[:, h_start:h_end, w_start:w_end]
        img_crop = img_crop.unsqueeze(0)
        img_crop = F.interpolate(img_crop, [int(h), int(w)], mode="bilinear", align_corners=False)
        return img_crop.squeeze(0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        # for zoom
        zoom_scale = random.random()
        if self.CZDA:
            do_zoom = self.is_train and zoom_scale > 0.5
        else:
            do_zoom = False
        if do_zoom:
            zoom_scale_ = random.uniform(0.5, 1.0)   

        line = self.filenames[index].split()
        if not self.is_train and self.gt_depth_path is not None:
            gt_depth = self.gt_depths[index]
            inputs['gt_depth'] = gt_depth

        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                try:
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                except:
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, side, do_flip)

        # adjusting intrinsics for resized img
        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
            # color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        # for zoom
        if do_zoom:
            # do zoom with CZDA
            # self.zoom_crop(inputs, zoom_scale)
            # K[0][0] = K[0][0] / zoom_scale
            # K[1][1] = K[1][1] / zoom_scale
            # zoom_ratio = zoom_scale
            self.zoom_crop_xy(inputs, zoom_scale, zoom_scale_)
            # change the fx and fy by scale
            K[0][0] = K[0][0] / zoom_scale  # fx
            K[1][1] = K[1][1] / zoom_scale_  # fy
        else:
            self.zoom_crop(inputs, 1.0)
            # zoom_ratio = 1.0

        # only for test post process
        if self.test:
            self.zoom_crop_test(inputs, scale=0.5, scale_twc=0.75)
            
        inv_K = np.linalg.pinv(K)
        inputs[("K")] = torch.from_numpy(K)
        inputs[("inv_K")] = torch.from_numpy(inv_K)
        # inputs[("zoom_scale")] = zoom_ratio

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]


        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.015
            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_pose(self, folder, frame_index, offset):
        return
