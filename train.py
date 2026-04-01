"""
Utility functions for the Flower Image Classifier project.
Project submission - Udacity
Part 2 - Training script

Author: AA
Date: 31.03.2026

This script trains a deep neural network on the Oxford 102 Flowers dataset and
saves a checkpoint for inference with predict.py.

Defaults are aligned with solution in
Part 1 notebook "Image Classifier Project.ipynb":
- arch: vgg16
- hidden_units: 512
- dropout: 0.5
- learning_rate: 0.001
- epochs: 5
- batch_size: 64
"""

import argparse
import os

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a flower image classifier"
    )

    # Basic usage: python train.py data_directory
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the flowers dataset directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Directory to save checkpoint (default: current directory)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg16",
        choices=["vgg16", "vgg13"],
        help='Choose architecture: "vgg16" or "vgg13" (default: vgg16)',
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="Hidden units in classifier (default: 512)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability (default: 0.5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs (default: 5)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for training if available",
    )

    return parser.parse_args()


def main():
    """Train the neural network and save a checkpoint."""
    args = parse_args()

    # Device selection
    device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    )

    # Data directories
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")

    # Define transforms for training and validation sets
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir,
                                      transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir,
                                      transform=data_transforms['valid']),
    }

    # Define dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'],
                                            batch_size=64,
                                            shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'],
                                            batch_size=64,
                                            shuffle=False),
    }

    # 1) Load a pretrained network (allow two architectures)
    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)

    # 2) Freeze parameters (feature extractor)
    for param in model.parameters():
        param.requires_grad = False

    # 3) Define a new classifier (ReLU + Dropout)
    input_features = model.classifier[0].in_features

    classifier = nn.Sequential(
        nn.Linear(input_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1),
    )

    model.classifier = classifier

    # 4) Loss + optimizer (only classifier params)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)

    model.to(device)

    # Training loop
    epochs = args.epochs
    print_every = 40
    steps = 0

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()

        for inputs, labels in dataloaders["train"]:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0.0
                accuracy = 0.0

                with torch.no_grad():
                    for v_inputs, v_labels in dataloaders['valid']:
                        v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

                        v_logps = model(v_inputs)
                        v_loss = criterion(v_logps, v_labels)
                        valid_loss += v_loss.item()

                        ps = torch.exp(v_logps)
                        preds = ps.argmax(dim=1)
                        accuracy += (preds == v_labels).float().mean().item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                    f"Valid accuracy: {accuracy/len(dataloaders['valid']):.3f}"
                )

                running_loss = 0.0
                model.train()

    # Save checkpoint
    model.class_to_idx = image_datasets["train"].class_to_idx

    checkpoint = {
        "arch": args.arch,
        "state_dict": model.state_dict(),
        "classifier": model.classifier,
        "class_to_idx": model.class_to_idx,
        "optimizer_state": optimizer.state_dict(),
        "epochs": epochs,
    }

    # Create the checkpoint directory if it does not already exist
    # os.makedirs works across operating systems and avoids errors if the folder exists
    os.makedirs(args.save_dir, exist_ok=True)

    
    # Build the full file path to the checkpoint in an OS-independent way
    # os.path.join ensures correct path separators on Windows / Linux / macOS
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")

    # Save the checkpoint dictionary to disk at the specified path
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()