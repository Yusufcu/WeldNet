# WeldNet

A lightweight deep learning model for welding defect detection and classification. WeldNet is designed to be efficient, accurate, and suitable for real-time industrial applications.

## Overview

WeldNet is a modern CNN architecture specifically designed for automated inspection of welding defects. It uses depthwise separable convolutions and attention mechanisms to achieve high accuracy with minimal computational requirements, making it ideal for deployment on edge devices and industrial systems.

### Key Features

- **Lightweight Architecture**: Significantly fewer parameters than traditional models (ResNet, VGG) while maintaining high accuracy
- **Real-time Performance**: Fast inference suitable for industrial applications
- **Attention Mechanisms**: Improves feature learning and model interpretability
- **Multiple Model Variants**: Small, standard, and large versions for different use cases
- **Easy to Use**: Simple training and inference scripts with comprehensive utilities

### Common Welding Defects Detected

- Porosity
- Cracks
- Slag inclusion
- Lack of fusion
- Undercut
- Good welds (non-defective)

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Yusufcu/WeldNet.git
cd WeldNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── img1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

Each subdirectory should represent a defect class (e.g., `good`, `crack`, `porosity`, etc.).

## Usage

### Training

Train a WeldNet model on your dataset:

```bash
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_type weldnet \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --output_dir outputs
```

#### Training Arguments

- `--train_dir`: Path to training data directory (required)
- `--val_dir`: Path to validation data directory (required)
- `--model_type`: Model architecture (`weldnet`, `weldnet_small`, `weldnet_large`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer to use (`adam`, `sgd`, `adamw`)
- `--scheduler`: Learning rate scheduler (`cosine`, `plateau`, `none`)
- `--input_size`: Input image size (default: 224)
- `--output_dir`: Directory to save outputs (default: outputs)
- `--early_stopping`: Enable early stopping
- `--patience`: Patience for early stopping (default: 15)
- `--resume`: Path to checkpoint to resume training

### Inference

Perform inference on single images or batches:

#### Single Image

```bash
python inference.py \
    --model_path outputs/best_model.pth \
    --input path/to/image.jpg \
    --class_names good,crack,porosity,slag,lack_of_fusion,undercut \
    --verbose
```

#### Batch Inference

```bash
python inference.py \
    --model_path outputs/best_model.pth \
    --input path/to/images/ \
    --class_names good,crack,porosity,slag,lack_of_fusion,undercut \
    --output results.csv
```

#### Inference Arguments

- `--model_path`: Path to trained model checkpoint (required)
- `--input`: Path to image or directory (required)
- `--model_type`: Model architecture used during training
- `--num_classes`: Number of classes (default: 6)
- `--class_names`: Comma-separated list of class names
- `--input_size`: Input image size (default: 224)
- `--verbose`: Print detailed probabilities
- `--output`: Path to save results CSV (for batch inference)

## Model Architecture

WeldNet uses a modern architecture combining:

1. **Depthwise Separable Convolutions**: Reduces parameters while maintaining representational power
2. **Residual Connections**: Enables training of deeper networks
3. **Attention Modules**: Improves feature learning and focuses on important regions
4. **Batch Normalization**: Stabilizes training and improves generalization

### Model Variants

- **WeldNet-Small**: Lightweight version with 0.5x width multiplier (~200K parameters)
- **WeldNet**: Standard version with 1.0x width multiplier (~800K parameters)
- **WeldNet-Large**: High-capacity version with 1.5x width multiplier (~1.8M parameters)

## Results

The model outputs include:

- **Training History Plot**: Loss and accuracy curves over epochs
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Precision, recall, and F1-score for each class
- **Model Checkpoints**: Saved model weights at best and final epochs

## Web Interface

WeldNet includes a web-based interface for easy demonstration and documentation:

```bash
cd web/templates
python -m http.server 8000
```

Then open http://localhost:8000 in your browser to access:
- Interactive demo with image upload
- Complete documentation
- Model comparison and specifications
- Usage examples

See `web/README.md` for more details on the web interface.

## Project Structure

```
WeldNet/
├── nets/
│   ├── __init__.py
│   └── weldnet.py          # Model architecture
├── utils/
│   ├── __init__.py
│   ├── dataset.py          # Dataset loading and preprocessing
│   └── train_utils.py      # Training utilities
├── web/                    # Web interface
│   ├── templates/          # HTML pages
│   ├── static/            # CSS and JavaScript
│   └── README.md          # Web interface docs
├── data/                   # Dataset directory
├── examples/               # Example scripts and notebooks
├── train.py               # Training script
├── inference.py           # Inference script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Performance Tips

### For Better Accuracy:
- Use data augmentation (enabled by default during training)
- Increase model capacity (use `weldnet_large`)
- Train for more epochs with early stopping
- Tune learning rate and batch size

### For Faster Inference:
- Use smaller model variant (`weldnet_small`)
- Reduce input image size
- Use GPU acceleration
- Consider model quantization for deployment

## Citation

If you use WeldNet in your research, please consider citing this repository:

```bibtex
@software{weldnet2024,
  title={WeldNet: A Lightweight Deep Learning Model for Welding Defect Detection},
  author={WeldNet Contributors},
  year={2024},
  url={https://github.com/Yusufcu/WeldNet}
}
```

For the original research on lightweight CNNs for welding defect detection, see:
- [WeldNet: a lightweight deep learning model for welding defect recognition](https://link.springer.com/article/10.1007/s40194-024-01759-9)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Inspired by research on lightweight CNN architectures for industrial inspection
- Built with PyTorch and related libraries
- Thanks to the welding inspection community for valuable feedback

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a research and educational project. For production deployment in critical applications, thorough testing and validation are required.
