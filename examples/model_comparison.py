"""
Example script demonstrating basic WeldNet usage

This script shows how to:
1. Load and inspect the model
2. Process a single image
3. Visualize model architecture
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from nets import weldnet, weldnet_small, weldnet_large


def print_model_info(model, model_name):
    """Print model information"""
    print(f"\n{'='*60}")
    print(f"{model_name} Information")
    print(f"{'='*60}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Measure inference time
    import time
    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    print(f"\nAverage inference time (CPU): {avg_time_ms:.2f} ms")
    print(f"Estimated FPS: {1000 / avg_time_ms:.1f}")


def main():
    """Main function"""
    print("WeldNet Model Comparison")
    print("=" * 60)
    
    # Create models
    num_classes = 6
    
    model_small = weldnet_small(num_classes=num_classes)
    model_standard = weldnet(num_classes=num_classes)
    model_large = weldnet_large(num_classes=num_classes)
    
    # Print information for each model
    print_model_info(model_small, "WeldNet-Small")
    print_model_info(model_standard, "WeldNet-Standard")
    print_model_info(model_large, "WeldNet-Large")
    
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print("\nUse WeldNet-Small for:")
    print("  - Edge devices with limited resources")
    print("  - Real-time applications requiring maximum speed")
    print("  - Mobile deployment")
    
    print("\nUse WeldNet-Standard for:")
    print("  - Balanced performance and accuracy")
    print("  - General industrial applications")
    print("  - Desktop/server deployment")
    
    print("\nUse WeldNet-Large for:")
    print("  - Maximum accuracy requirements")
    print("  - Complex defect patterns")
    print("  - Research and development")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
