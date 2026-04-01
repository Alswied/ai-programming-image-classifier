"""
Flower Image Classifier Project (Oxford 102 Flowers)
Project submission - Udacity
Part 2 - Training script

Author: AA
Date: 31.03.2026


This script loads a trained image classification model checkpoint and
predicts the most likely flower class for a given input image.

Basic usage:
    python predict.py /path/to/image checkpoint.pth

Optional arguments:
    --top_k K               Return the top K most likely classes (default: 5)
    --category_names FILE   JSON file mapping class labels to flower names
                            (default: cat_to_name.json)
    --gpu                   Use GPU for inference if available

The script outputs the predicted flower name(s) along with their
corresponding probabilities.
"""

import argparse
import json
import torch

from utils import load_checkpoint, predict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image"
    )

    parser.add_argument("image_path", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
    )
    parser.add_argument("--gpu", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    )

    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint, device)

    probs, classes = predict(
        args.image_path, model, device, topk=args.top_k
    )

    names = [cat_to_name[c] for c in classes]

    for i in range(len(probs)):
        print(f"{i + 1}: {names[i]} ({probs[i]:.3f})")


if __name__ == "__main__":
    main()