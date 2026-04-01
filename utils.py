# utils.py

"""
Utility functions for the Flower Image Classifier project.
Project submission - Udacity

Author: AA
Date: 31.03.2026

This module provides reusable helpers for image preprocessing, checkpoint
loading, and inference. It is shared by train.py and predict.py.
"""
import torch
import numpy as np
from PIL import Image
from torchvision import models

def process_image(image):
    """
    Preprocess a PIL image for use in a PyTorch model.

    The image is resized so the shortest side is 256 pixels, center‑cropped
    to 224x224, normalized using ImageNet statistics, and converted to
    channel‑first format.

    Args:
        image (PIL.Image): Input image.

    Returns:
        np.ndarray: Preprocessed image array with shape (3, 224, 224).
    """

    # Get original image dimensions (width, height)
    width, height = image.size

    # Resize so that the shortest side is 256 pixels
    # while keeping the aspect ratio
    if width < height:
        new_width = 256
        new_height = int(height * (256 / width))
    else:
        new_height = 256
        new_width = int(width * (256 / height))

    # Resize the image to the new dimensions
    image = image.resize((new_width, new_height))

    # Calculate coordinates for center cropping to 224x224
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224

    # Crop the image around the center
    image = image.crop((left, top, right, bottom))

    # Convert PIL image to NumPy array,
    # cast to float, and scale pixel values from [0, 255] → [0, 1]
    np_image = np.array(image).astype(np.float32) / 255.0

    # ImageNet mean values used during model training
    mean = np.array([0.485, 0.456, 0.406])

    # ImageNet standard deviation values used during model training
    std = np.array([0.229, 0.224, 0.225])

    # Normalize the image: (pixel - mean) / std
    np_image = (np_image - mean) / std

    # Reorder dimensions from (H, W, C) → (C, H, W) - C:color, H: height , W: width
    # because PyTorch expects channels first
    np_image = np_image.transpose((2, 0, 1))

    # Return the processed image as a NumPy array
    return np_image


def load_checkpoint(filepath, device):
    """
    Load a trained model checkpoint and rebuild the network.

    Args:
        filepath (str): Path to the checkpoint file.
        device (torch.device): CPU or CUDA device.

    
    Returns:
        torch.nn.Module: Reconstructed model in evaluation mode.
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Rebuild the pretrained model architecture used during training
    arch = checkpoint["arch"]

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    model.to(device)
    model.eval()
    return model


def predict(image_path, model, device, topk=5):
    """
    Predict the top‑K classes of an image using a trained deep learning model.

    Args:
        image_path (str): Path to the input image.
        model (torch.nn.Module): Trained model.
        device (torch.device): CPU or CUDA device.
        topk (int): Number of top predictions to return.

    Returns:
        tuple[np.ndarray, list[str]]:
            - Probabilities of the top‑K predictions
            - Corresponding class labels
    """

    # Load image from disk and force RGB format
    # (some images may be grayscale or RGBA; the model expects 3 channels)
    image = Image.open(image_path).convert("RGB")

    # Apply the exact same preprocessing used during training:
    # resize → center crop → normalize → channel reorder
    np_image = process_image(image)

    # Convert NumPy array to PyTorch tensor
    # - unsqueeze(0) adds a batch dimension: (3,224,224) → (1,3,224,224)
    # - float() ensures correct dtype
    # - move tensor to the same device as the model (CPU or GPU)
    image_tensor = torch.from_numpy(np_image).unsqueeze(0).float().to(device)

    # Switch the model to evaluation mode:
    # - disables dropout
    # - uses running statistics for batch normalization
    model.eval()

    # Disable gradient tracking:
    # - saves memory
    # - speeds up inference
    # - prevents accidental backpropagation
    with torch.no_grad():

        # Forward pass:
        # pass the preprocessed image through the network
        # output is log-probabilities because the model ends with LogSoftmax
        logps = model(image_tensor)

        # Convert log-probabilities → probabilities
        # exp(log(p)) = p
        ps = torch.exp(logps)

        # Extract the top-K probabilities and their corresponding class indices
        # - top_p: highest K probabilities
        # - top_idx: indices of those probabilities in the output tensor
        top_p, top_idx = ps.topk(topk, dim=1)

    # Remove the batch dimension (size 1) and move tensors to CPU
    # NumPy conversion is required for printing and plotting
    # squeeze(0) removes batch dimension (1, K) → (K,)
    top_p = top_p.squeeze(0).cpu().numpy()
    top_idx = top_idx.squeeze(0).cpu().numpy()

    # Convert predicted indices back to original class labels:
    # - model outputs indices (0–101)
    # - class_to_idx maps class label → index
    # - we invert it to map index → class label
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_idx]

    # Return:
    # - top_p: probabilities of the top-K predictions
    # - top_classes: corresponding class labels (e.g. flower IDs)
    return top_p, top_classes
