#!/usr/bin/env python3
# src/test_here.py
# ------------------------------------------------------------
# Custom CRAFT-pytorch detector:
#   • Accepts either a single image or a folder of images
#   • Runs character‐region & affinity detection with tweakable params
#   • Saves heatmap masks and overlayed boxes to disk
#
# Based on test.py from the official CRAFT-pytorch repo (MIT License)
# ------------------------------------------------------------

import os
import sys
import time
import argparse
from collections import OrderedDict

# Make sure we import the local CRAFT-pytorch code
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '..', 'CRAFT-pytorch'))
sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import craft_utils
import imgproc
import file_utils
from craft import CRAFT


def copyStateDict(state_dict):
    """Strip DataParallel 'module.' prefix if present."""
    items = list(state_dict.items())
    if items[0][0].startswith("module"):
        start = 1
    else:
        start = 0
    new = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start:])
        new[name] = v
    return new


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(net, image, cfg, refine_net=None):
    """
    Run one forward pass + post‐processing of CRAFT on `image`.
    Returns: (boxes, polys, heatmap_img)
    """
    t0 = time.time()

    # 1) resize & normalize
    img_resized, ratio, _ = imgproc.resize_aspect_ratio(
        image, cfg.canvas_size,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=cfg.mag_ratio
    )
    rw = rh = 1 / ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
    if cfg.cuda:
        x = x.cuda()

    # 2) forward
    with torch.no_grad():
        y, feature = net(x)
    score_text = y[0,:,:,0].cpu().numpy()
    score_link = y[0,:,:,1].cpu().numpy()

    # 3) optional refiner
    if refine_net:
        with torch.no_grad():
            y_ref = refine_net(y, feature)
        score_link = y_ref[0,:,:,0].cpu().numpy()

    # 4) **get boxes & polys from the raw score maps**
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link,
        cfg.text_threshold,
        cfg.link_threshold,
        cfg.low_text,
        cfg.poly
    )

    # 5) adjust boxes back to original scale
    boxes = craft_utils.adjustResultCoordinates(boxes, rw, rh)

    # 6) manually scale polygons
    scaled_polys = []
    for i, p in enumerate(polys):
        if p is None:
            # fallback: turn [x1,y1,x2,y2] into a 4-point box
            x1, y1, x2, y2 = boxes[i]
            poly = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ], dtype=np.int32)
        else:
            arr = np.array(p, dtype=np.float32)
            poly = (arr * np.array([rw, rh])).astype(np.int32)
        scaled_polys.append(poly)
    polys = scaled_polys


    # 7) render combined heatmap for saving/display
    render = np.hstack((score_text, score_link))
    heatmap_img = imgproc.cvt2HeatmapImg(render)

    return boxes, polys, heatmap_img



def main():
    parser = argparse.ArgumentParser(
        description="Custom CRAFT-pytorch detector (file or folder)"
    )
    # model + device
    parser.add_argument('--trained_model',
                        default='weights/craft_mlt_25k.pth',
                        help="Path to CRAFT .pth weights")
    parser.add_argument('--refiner_model',
                        default='weights/craft_refiner_CTW1500.pth',
                        help="Path to LinkRefiner .pth weights (optional)")
    parser.add_argument('--cuda', type=str2bool, default=True,
                        help="Use CUDA for inference")
    # detection thresholds
    parser.add_argument('--text_threshold',  type=float, default=0.5,
                        help="Character score threshold")
    parser.add_argument('--low_text',        type=float, default=0.25,
                        help="Low-bound text threshold")
    parser.add_argument('--link_threshold',  type=float, default=0.25,
                        help="Affinity threshold")
    # image preprocessing
    parser.add_argument('--canvas_size',     type=int,   default=3000,
                        help="Longest image side after resize")
    parser.add_argument('--mag_ratio',       type=float, default=1.,
                        help="Additional up-sampling ratio")
    # box type & timing
    parser.add_argument('--poly', action='store_true', default=False,
                        help="Enable polygon output")
    parser.add_argument('--refine', action='store_true', default=False,
                        help="Use link refiner network")
    parser.add_argument('--show_time', action='store_true', default=False,
                        help="Print timing per image")
    # I/O
    parser.add_argument('input', help="Image file or folder of images")
    parser.add_argument('--output_folder', default='./result/',
                        help="Where to save masks & results")
    args = parser.parse_args()

    # prepare output
    os.makedirs(args.output_folder, exist_ok=True)

    # 1) load CRAFT
    net = CRAFT()
    # auto-disable CUDA if it’s not actually available
    if args.cuda and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available – falling back to CPU")
        args.cuda = False

    # load weights
    print(f"Loading weights: {args.trained_model}")
    map_loc = 'cpu' if not args.cuda else None
    state = torch.load(args.trained_model, map_location=map_loc)
    state = torch.load(args.trained_model,
                       map_location='cpu' if not args.cuda else None)
    net.load_state_dict(copyStateDict(state))
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()

    # 2) optional refiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print(f"Loading refiner: {args.refiner_model}")
        state = torch.load(args.refiner_model,
                           map_location='cpu' if not args.cuda else None)
        refine_net.load_state_dict(copyStateDict(state))
        if args.cuda:
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        refine_net.eval()
        args.poly = True

    # 3) build list of images
    if os.path.isdir(args.input):
        image_list, _, _ = file_utils.get_files(args.input)
    elif os.path.isfile(args.input):
        image_list = [args.input]
    else:
        print(f"ERROR: '{args.input}' is not a file or folder.")
        sys.exit(1)

    # 4) process each image
    t0 = time.time()
    for i, img_path in enumerate(image_list, 1):
        print(f"[{i}/{len(image_list)}] {img_path}", end='\r')
        image = imgproc.loadImage(img_path)
        boxes, polys, heatmap = test_net(net, image, args, refine_net)

        # save heatmap
        base = os.path.splitext(os.path.basename(img_path))[0]
        heat_fn = os.path.join(args.output_folder, f"{base}_heat.jpg")
        cv2.imwrite(heat_fn, heatmap)

        # save detection overlays + JSON
        file_utils.saveResult(
            img_path, image[:, :, ::-1], polys,
            dirname=args.output_folder
        )

    print(f"\nDone in {time.time() - t0:.1f}s")

if __name__ == '__main__':
    main()
