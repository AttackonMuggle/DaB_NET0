from __future__ import absolute_import, division, print_function
import cv2
import sys
import numpy as np

import paddle
from paddle.io import DataLoader
import paddle.nn.functional as F
import os

sys.path.append('.')
from dataset.kitti_dataset import KITTIRAWDataset
from pd_model_trace.x2paddle_code import dab_net
from utils import disp_to_depth, readlines, compute_errors


MIN_DEPTH=1e-3
MAX_DEPTH=80
STEREO_SCALE_FACTOR = 36

cv2.setNumThreads(0)
zoom_post_process = False
use_stereo_scale = False

def evaluate(MODEL_PATH, DATA_PATH, GT_PATH):
    filenames = readlines("./dataset/splits/test/test_files.txt")

    dataset = KITTIRAWDataset(DATA_PATH,
                              filenames,
                              320,
                              1024,
                              [0],
                              is_train=False,
                              img_ext='.png',
                              SSF_detacher=True,
                              test=True)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False)  # shuffle=True disorganize the data list

    # test for zoom-pp
    paddle.disable_static()
    params = paddle.load(MODEL_PATH)
    model = dab_net()
    model.set_dict(params, use_structured_name=True)
    model.eval()


    pred_disps = []

    with paddle.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            disp = model(inputs["color_aug", 0, 0])

            if zoom_post_process:
                disp = zoom_pp(inputs, disp, model)
                disp = zoom_pp_twice(inputs, disp, model, scale=0.75)
            pred_disp, _ = disp_to_depth(disp, 0.1, 100)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            # pred_disp = pred_disp[:, 0].numpy()
            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    gt_depths = np.load(GT_PATH, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")
    if use_stereo_scale:
        print('using baseline')
    else:
        print('using mean scaling')

    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1

        mask = np.logical_and(mask, crop_mask)

        pred_depth_ = pred_depth[mask]  # [16985]
        gt_depth_ = gt_depth[mask]

        ratio = np.median(gt_depth_) / np.median(pred_depth_)
        ratios.append(ratio)

        if use_stereo_scale:
            ratio = STEREO_SCALE_FACTOR

        pred_depth_ *= ratio
        pred_depth_[pred_depth_ < MIN_DEPTH] = MIN_DEPTH
        pred_depth_[pred_depth_ > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth_, pred_depth_))

    ratios = np.array(ratios)
    med = np.median(ratios)
    mean_errors = np.array(errors).mean(0)
    print("Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    print("\n" + ("{:>}| " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{:.3f} " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def zoom_pp(inputs, disp, model, disp_grid=None, scale=0.5):
    _, _, H, W = disp.shape
    h = H * scale
    w = W * scale
    disp_dic = model(inputs[("color_zoom_pp", 0, 0)])
    disp_zoom = F.interpolate(disp_dic, [int(h), int(w)], mode='bilinear', align_corners=False)

    # center crop
    h_start = int((H - h) / 2)
    h_end = int(H - h_start)
    w_start = int((W - w) / 2)
    w_end = int(W - w_start)
    disp_crop = disp[:, :, h_start:h_end, w_start:w_end]

    # average
    disp_avg = (disp_crop + disp_zoom) / 2
    disp[:, :, h_start:h_end, w_start:w_end] = disp_avg

    return disp

def zoom_pp_twice(inputs, disp, model, disp_grid=None, scale=0.75):
    _, _, H, W = disp.shape
    h = H * scale
    w = W * scale
    disp_dic = model(inputs[("color_zoom_pp_twice", 0, 0)])
    disp_zoom = F.interpolate(disp_dic, [int(h), int(w)], mode='bilinear', align_corners=False)

    # center crop
    h_start = int((H - h) / 2)
    h_end = int(H - h_start)
    w_start = int((W - w) / 2)
    w_end = int(W - w_start)
    disp_crop = disp[:, :, h_start:h_end, w_start:w_end]

    # average
    disp_avg = (disp_crop + disp_zoom) / 2
    disp[:, :, h_start:h_end, w_start:w_end] = disp_avg

    return disp

def crop_size(img, scale=0.5):
    H, W = img.shape
    h = H * scale
    w = W * scale

    h_start = int((H - h) / 2)
    h_end = int(H - h_start)
    w_start = int((W - w) / 2)
    w_end = int(W - w_start)
    img_crop = img[h_start:h_end, w_start:w_end]

    return img_crop
    

if __name__ == "__main__":
    DATA_PATH = '../Kitti'#path to cfg file
    GT_PATH = './dataset/splits/gt_depths.npz'#path to kitti gt depth
    MODEL_PATH = './pd_model_trace/model.pdparams'#path to model weights

    evaluate(MODEL_PATH, DATA_PATH, GT_PATH)
