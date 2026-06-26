"""
Deblur an image using NAFNet pre-trained models (GoPro or REDS).
Usage: python deblur_image.py --input <input_path> --output <output_path> [--model gopro|reds]
"""
import sys
import os
import argparse

# Add local basicsr to path before importing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import cv2
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse


def imread(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)


def single_image_inference(model, img, save_path):
    model.feed_data(data={"lq": img.unsqueeze(dim=0)})

    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals["result"]])
    imwrite(sr_img, save_path)
    print(f"Deblurred image saved to: {save_path}")
    return sr_img


def main():
    parser = argparse.ArgumentParser(description="Deblur an image using NAFNet")
    parser.add_argument("--input", type=str, required=True, help="Path to input blurry image")
    parser.add_argument("--output", type=str, default="output/output.png", help="Path to save deblurred image")
    parser.add_argument("--model", type=str, choices=["gopro", "reds"], default="reds",
                        help="Model: gopro (motion blur) or reds (blur + JPEG artifacts)")
    args = parser.parse_args()

    # Select config and model
    if args.model == "gopro":
        opt_path = "options/test/GoPro/NAFNet-width64.yml"
        model_path = "experiments/pretrained_models/NAFNet-GoPro-width64.pth"
        print("Using GoPro model (motion blur)")
    else:
        opt_path = "options/test/REDS/NAFNet-width64.yml"
        model_path = "experiments/pretrained_models/NAFNet-REDS-width64.pth"
        print("Using REDS model (blur + JPEG artifacts)")

    # Check model exists
    if not os.path.exists(model_path):
        alt_path = f"experiments/pretrained_models/NAFNet-{args.model.upper()}-width64.pth"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run: gdown 'https://drive.google.com/...' -O '{model_path}'"
            )

    # Parse options
    opt = parse(opt_path, is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 0  # CPU mode
    opt["path"]["pretrain_network_g"] = model_path

    # Read input image
    print(f"Reading input image: {args.input}")
    img_input = imread(args.input)
    print(f"Image shape: {img_input.shape}")

    # Convert to tensor
    inp = img2tensor(img_input)

    # Create model and run inference
    print("Creating model and loading weights...")
    model = create_model(opt)

    print("Running deblur inference...")
    result = single_image_inference(model, inp, args.output)

    # Calculate difference
    diff = np.abs(img_input.astype(np.float32) - result.astype(np.float32))
    mean_diff = np.mean(diff)
    pct_changed = 100 * np.sum(diff > 5) / diff.size
    print(f"\n--- Result Statistics ---")
    print(f"Mean absolute pixel change: {mean_diff:.2f} / 255")
    print(f"Pixels changed > 5: {pct_changed:.1f}%")
    print(f"Done!")


if __name__ == "__main__":
    main()