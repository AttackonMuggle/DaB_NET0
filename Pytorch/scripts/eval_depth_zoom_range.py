from __future__ import absolute_import, division, print_function
import cv2
import sys
import numpy as np
from mmcv import Config

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

import PIL.Image as pil
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

sys.path.append('.')
from mono.model.registry import MONO
from mono.model.mono_resnet.layers import disp_to_depth
from mono.datasets.utils import readlines, compute_errors
from mono.datasets.kitti_dataset import KITTIRAWDataset
from mono.model.mono_hrnet.hr_net import hrnet18
from mono.model.mono_hrnet.depth_decoder import DepthDecoder


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
STEREO_SCALE_FACTOR = 36
MIN_DEPTH=1e-3
MAX_DEPTH=80

save_error = False
do_eval_on_zoom_range = True
do_eval_on_crop_range = False
save_path = os.path.join("./visulization/")

def evaluate(MODEL_PATH, CFG_PATH, GT_PATH):
    filenames = readlines("./mono/datasets/splits/test/test_files.txt")
    cfg = Config.fromfile(CFG_PATH)

    dataset = KITTIRAWDataset(cfg.data['in_path'],
                              filenames,
                              cfg.data['height'],
                              cfg.data['width'],
                              [0],
                              is_train=False,
                              img_ext='.png',
                              gt_depth_path=GT_PATH,
                              CZDA=True,
                              test=True)

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=False)  # shuffle=True disorganize the data list

    # test for zoom-pp
    depth_encoder = hrnet18()
    depth_decoder = DepthDecoder([ 64, 18, 36, 72, 144 ])
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    for name, param in depth_encoder.state_dict().items():
        depth_encoder.state_dict()[name].copy_(checkpoint['state_dict']['DepthEncoder.' + name])
    for name, param in depth_decoder.state_dict().items():
        depth_decoder.state_dict()[name].copy_(checkpoint['state_dict']['DepthDecoder.' + name])
    depth_encoder.cuda()
    depth_encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()


    pred_disps = []
    colors = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            color = inputs["color", 0, 0]
            outputs = depth_decoder(depth_encoder(inputs["color_aug", 0, 0]))

            disp = outputs[("disp", 0, 0)]

            if do_eval_on_zoom_range:
                disp = zoom_pp_zoom_range(inputs, disp, depth_encoder, depth_decoder)
                    
            pred_disp, _ = disp_to_depth(disp, 0.1, 100)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            color = color.cpu().numpy()
            pred_disps.append(pred_disp)
            colors.append(color)

    pred_disps = np.concatenate(pred_disps)
    colors = np.concatenate(colors)
    
    gt_depths = np.load(GT_PATH, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")
    if cfg.data['stereo_scale']:
        print('using baseline')
    else:
        print('using mean scaling')

    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):
        gt_depth_org = gt_depths[i]
        gt_org_h, gt_org_w = gt_depth_org.shape[:2]
        if do_eval_on_zoom_range:
            gt_depth = crop_size(gt_depth_org, scale=0.5)
        elif do_eval_on_crop_range:
            gt_depth = crop_size(gt_depth_org, scale=0.5)
        else:
            gt_depth = gt_depth_org
        gt_height, gt_width = gt_depth.shape[:2]

        color = colors[i].transpose(1,2,0)
        color = cv2.resize(color, (gt_org_w, gt_org_h))
        
        pred_disp = pred_disps[i]
        if do_eval_on_crop_range:
            pred_disp = crop_size(pred_disp, scale=0.5)
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        if do_eval_on_zoom_range or do_eval_on_crop_range:
            # get mask
            crop_zoom = np.array([0.40810811 * gt_org_h, 0.99189189 * gt_org_h,
                         0.03594771 * gt_org_w,  0.96405229 * gt_org_w]).astype(np.int32)
            crop_mask_pp = np.zeros(gt_depth_org.shape)   # [375, 1242]
            crop_mask_pp[crop_zoom[0]:crop_zoom[1], crop_zoom[2]:crop_zoom[3]] = 1
            crop_mask_zoom = crop_size(crop_mask_pp, scale=0.5)  # [189, 622]
            mask_zoom = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)  # [189, 622]
            mask_zoom = np.logical_and(mask_zoom, crop_mask_zoom)
            # get ratio
            pred_depth_ = pred_depth[mask_zoom]
            gt_depth_ = gt_depth[mask_zoom]
            ratio = np.median(gt_depth_) / np.median(pred_depth_)
        else:
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

        if cfg.data['stereo_scale']:
            ratio = STEREO_SCALE_FACTOR
        
        # generate error map
        if save_error:
            save_disp(pred_disp, color, save_path, i)

        pred_depth_ *= ratio
        pred_depth_[pred_depth_ < MIN_DEPTH] = MIN_DEPTH
        pred_depth_[pred_depth_ > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth_, pred_depth_))

    ratios = np.array(ratios)
    med = np.median(ratios)
    mean_errors = np.array(errors).mean(0)
    print("Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    print("\n" + ("{:>}| " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{:.5f} " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def save_disp(pred_disp, color, save_path, iter_img):
    save_path_disp = save_path + "disp/"
    os.makedirs(save_path_disp, exist_ok=True)
    save_path_color = save_path + "color/"
    os.makedirs(save_path_color, exist_ok=True)
    save_disp_name = save_path_disp + "disp{}.png".format(iter_img+1)
    save_color_name = save_path_color + "color{}.png".format(iter_img+1)

    # save disp grey
    # disp = 255.0 * (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
    # cv2.imwrite(save_disp_name, disp.astype('uint8'))

    # save disp color
    disp = pred_disp
    vmax = 6.0
    normalizer = mpl.colors.Normalize(vmin=0.01, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.save(save_disp_name)

    # save color RGB
    color = 255.0 * color
    color_BGR = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    color_BGR = np.clip(color_BGR, 0, 255)
    cv2.imwrite(save_color_name, color_BGR.astype('uint8'))
    

def crop_size(img, scale=0.5):
    H, W = img.shape
    h = H * scale
    w = W * scale

    h_start = int((H - h) / 2)
    h_end = int(H - h_start)
    # w_start = int(w)
    # img_crop = img[h_start:h_end, w_start:]
    w_start = int((W - w) / 2)
    w_end = int(W - w_start)
    img_crop = img[h_start:h_end, w_start:w_end]

    return img_crop

def zoom_pp_zoom_range(inputs, disp, encoder, decoder, disp_grid=None, scale=0.5, do_fuse=False):
    _, _, H, W = disp.shape
    h = H * scale
    w = W * scale
    disp_dic = decoder(encoder(inputs[("color_zoom_pp", 0, 0)]))
    disp_zoom = disp_dic[("disp", 0, 0)]
    # disp_zoom = F.interpolate(disp_zoom, [int(h), int(w)], mode='bilinear', align_corners=False)
    if do_fuse:
        # center crop
        h_start = int((H - h) / 2)
        h_end = int(H - h_start)
        w_start = int((W - w) / 2)
        w_end = int(W - w_start)
        disp_zoom = F.interpolate(disp_zoom, [int(h), int(w)], mode='bilinear', align_corners=False)
        disp_crop = disp[:, :, h_start:h_end, w_start:w_end]
        disp_avg = (disp_crop + disp_zoom) / 2
        
        return disp_avg
    else:
        return disp_zoom
    

if __name__ == "__main__":
    CFG_PATH = './config/cfg_kitti_hrnet.py'#path to cfg file
    GT_PATH = './pre_trained/gt_depths.npz'#path to kitti gt depth
    MODEL_PATH = './re_1s_zoom_diff_qkvmut2_M.pth'#path to model weights

    evaluate(MODEL_PATH, CFG_PATH, GT_PATH)