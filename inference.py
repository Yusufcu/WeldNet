"""
Inference script for WeldNet

This script performs inference on welding defect images using a trained model.
"""

import os
import argparse
import torch
from PIL import Image
import numpy as np

from nets import weldnet, weldnet_small, weldnet_large
from utils import get_transforms


def get_model(model_type, num_classes, in_channels):
    """Get model based on type"""
    models = {
        'weldnet': weldnet,
        'weldnet_small': weldnet_small,
        'weldnet_large': weldnet_large
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](num_classes=num_classes, in_channels=in_channels)


def load_model(model_path, model_type, num_classes, in_channels, device):
    """Load trained model from checkpoint"""
    model = get_model(model_type, num_classes, in_channels)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_image(model, image_path, transform, device, class_names):
    """Predict defect class for a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    
    # Get all class probabilities
    all_probs = probabilities.cpu().numpy()[0]
    
    return predicted_class, confidence, all_probs


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Parse class names
    class_names = args.class_names.split(',') if args.class_names else None
    num_classes = len(class_names) if class_names else args.num_classes
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.model_type, num_classes, args.in_channels, device)
    print("Model loaded successfully")
    
    # Get transform
    transform = get_transforms(input_size=args.input_size, augment=False)
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        print(f"\nProcessing image: {args.input}")
        predicted_class, confidence, all_probs = predict_image(
            model, args.input, transform, device, class_names
        )
        
        print(f"\nPrediction Results:")
        print(f"Predicted Class: {class_names[predicted_class] if class_names else predicted_class}")
        print(f"Confidence: {confidence*100:.2f}%")
        
        if args.verbose:
            print("\nAll Class Probabilities:")
            for i, prob in enumerate(all_probs):
                class_name = class_names[i] if class_names else f"Class {i}"
                print(f"  {class_name}: {prob*100:.2f}%")
    
    elif os.path.isdir(args.input):
        # Directory of images
        print(f"\nProcessing directory: {args.input}")
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print("No image files found in directory")
            return
        
        results = []
        for img_file in image_files:
            img_path = os.path.join(args.input, img_file)
            predicted_class, confidence, all_probs = predict_image(
                model, img_path, transform, device, class_names
            )
            results.append({
                'file': img_file,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': all_probs
            })
        
        # Print results
        print(f"\nProcessed {len(results)} images:")
        print(f"{'File':<40} {'Predicted Class':<20} {'Confidence':<12}")
        print("-" * 72)
        for result in results:
            class_name = class_names[result['predicted_class']] if class_names else f"Class {result['predicted_class']}"
            print(f"{result['file']:<40} {class_name:<20} {result['confidence']*100:>6.2f}%")
        
        # Save results to CSV if requested
        if args.output:
            import csv
            with open(args.output, 'w', newline='') as csvfile:
                fieldnames = ['file', 'predicted_class', 'confidence']
                if class_names:
                    fieldnames.extend(class_names)
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        'file': result['file'],
                        'predicted_class': class_names[result['predicted_class']] if class_names else result['predicted_class'],
                        'confidence': f"{result['confidence']:.4f}"
                    }
                    if class_names:
                        for i, class_name in enumerate(class_names):
                            row[class_name] = f"{result['probabilities'][i]:.4f}"
                    writer.writerow(row)
            
            print(f"\nResults saved to {args.output}")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with WeldNet for welding defect classification')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory of images')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='weldnet',
                       choices=['weldnet', 'weldnet_small', 'weldnet_large'],
                       help='Model architecture (default: weldnet)')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of classes (default: 6)')
    parser.add_argument('--in_channels', type=int, default=3,
                       help='Number of input channels (default: 3)')
    parser.add_argument('--class_names', type=str, default=None,
                       help='Comma-separated list of class names (e.g., "good,crack,porosity")')
    
    # Inference parameters
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA inference')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed probabilities for all classes')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results CSV (for directory input)')
    
    args = parser.parse_args()
    main(args)
