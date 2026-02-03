"""
Dataset utilities for WeldNet

This module provides dataset loaders and preprocessing for welding defect images.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class WeldingDefectDataset(Dataset):
    """
    Dataset class for welding defect images
    
    Expected directory structure:
        root/
            class1/
                img1.jpg
                img2.jpg
                ...
            class2/
                img1.jpg
                img2.jpg
                ...
    
    Args:
        root_dir (str): Root directory of the dataset
        transform (callable, optional): Optional transform to be applied on images
        class_names (list, optional): List of class names. If None, will be inferred from directory names.
    """
    
    def __init__(self, root_dir, transform=None, class_names=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get class names from subdirectories
        if class_names is None:
            self.class_names = sorted([d for d in os.listdir(root_dir) 
                                      if os.path.isdir(os.path.join(root_dir, d))])
        else:
            self.class_names = class_names
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(input_size=224, augment=True):
    """
    Get data transforms for training and validation
    
    Args:
        input_size (int): Size of input images
        augment (bool): Whether to apply data augmentation (for training)
    
    Returns:
        transform: Composed transforms
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(train_dir, val_dir, batch_size=32, input_size=224, num_workers=4):
    """
    Create train and validation dataloaders
    
    Args:
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        batch_size (int): Batch size for training
        input_size (int): Size of input images
        num_workers (int): Number of workers for data loading
    
    Returns:
        train_loader, val_loader: DataLoader objects
        num_classes: Number of classes in the dataset
        class_names: List of class names
    """
    # Create transforms
    train_transform = get_transforms(input_size=input_size, augment=True)
    val_transform = get_transforms(input_size=input_size, augment=False)
    
    # Create datasets
    train_dataset = WeldingDefectDataset(train_dir, transform=train_transform)
    val_dataset = WeldingDefectDataset(val_dir, transform=val_transform,
                                      class_names=train_dataset.class_names)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.class_names)
    class_names = train_dataset.class_names
    
    return train_loader, val_loader, num_classes, class_names


if __name__ == '__main__':
    # Test dataset loading
    print("WeldingDefectDataset module test")
    print("To test, provide train and validation directories with the expected structure")
