# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np

from dgecn import dgecn
import torch
from torchvision import transforms, datasets

from layers import disp_to_depth
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images',
                        default='./assert')
    parser.add_argument('--model_path', type=str,
                        help='name of a pretrained model to use',
                        default='dgecn.pth')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')


    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    data_cfg = './data/data-YCB.cfg'
    data_options = read_data_cfg(data_cfg)
    # LOADING PRETRAINED MODEL
    print("   Loading pretrained dgecn")
    model = dgecn(data_options)
    model = torch.load(args.model_path)
    model.to(device)
    model.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            front, _ = os.path.splitext(image_path)
            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue
            start = time.time()
            # Load image and preprocess
            img = cv2.imread(image_path)
            colormapped_im = do_detect_depth(model, img, use_gpu=True)
            name_dest_im = "{}_disp.jpeg".format(front)
            print("outdir: ", name_dest_im)
            cv2.imwrite(name_dest_im, colormapped_im)
            #im.save(name_dest_im)
            finish = time.time()
            print('%s: Visualization in %f seconds.' % (image_path, (finish - start)))


    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
