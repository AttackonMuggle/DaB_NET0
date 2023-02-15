from __future__ import absolute_import, division, print_function
import sys
import numpy as np

import paddle
import os

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from paddle.vision import transforms

sys.path.append('.')
from pd_model_trace.x2paddle_code import dab_net


def main(MODEL_PATH, image_path):
    feed_width =  1024
    feed_height = 320

    # x2paddle model
    paddle.disable_static()
    params = paddle.load(MODEL_PATH)
    model = dab_net()
    model.set_dict(params, use_structured_name=True)
    model.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(image_path):
        # Only testing on a single image
        paths = [image_path]
        output_directory = os.path.dirname(image_path)
    else:
        raise Exception("Can not find image_path: {}".format(image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with paddle.no_grad():
        for idx, image_path in enumerate(paths):
            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)

            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            disp = model(input_image)

            disp_resized = paddle.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().numpy()
            print(disp_resized_np.shape)
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            name_dest_im = os.path.join(output_directory, "{}_disp.png".format(image_name))

            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))
    print('-> Done!')


if __name__ == "__main__":
    MODEL_PATH = './pd_model_trace/model.pdparams'#path to model weights
    image_path = './assets/test.png'
    
    main(MODEL_PATH, image_path)