"""
WeldNet: A lightweight deep learning model for welding defect detection

This module implements the WeldNet architecture, which is designed to be
efficient and accurate for real-time welding defect classification.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution for efficient feature extraction"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AttentionModule(nn.Module):
    """Lightweight attention mechanism for feature refinement"""
    
    def __init__(self, channels, reduction=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WeldNetBlock(nn.Module):
    """Basic building block for WeldNet with optional attention"""
    
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False):
        super(WeldNetBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = DepthwiseSeparableConv(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_attention:
            out = self.attention(out)
        out += self.shortcut(x)
        return out


class WeldNet(nn.Module):
    """
    WeldNet: Lightweight CNN for welding defect classification
    
    Args:
        num_classes (int): Number of defect classes to classify
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        width_multiplier (float): Width multiplier for network channels
    """
    
    def __init__(self, num_classes=6, in_channels=3, width_multiplier=1.0):
        super(WeldNet, self).__init__()
        
        # Calculate channel sizes based on width multiplier
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        base_channels = [32, 64, 128, 256]
        channels = [_make_divisible(c * width_multiplier) for c in base_channels]
        
        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=2,
                     padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction blocks
        self.layer1 = self._make_layer(channels[0], channels[0], 2, stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, stride=2, use_attention=True)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, stride=2, use_attention=True)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(channels[3], num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, use_attention=False):
        """Create a layer with multiple WeldNet blocks"""
        layers = []
        layers.append(WeldNetBlock(in_channels, out_channels, stride, use_attention))
        for _ in range(1, num_blocks):
            layers.append(WeldNetBlock(out_channels, out_channels, 1, use_attention))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def weldnet(num_classes=6, in_channels=3, pretrained=False):
    """
    Create a WeldNet model
    
    Args:
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels
        pretrained (bool): Load pretrained weights (not implemented yet)
    
    Returns:
        model: WeldNet model instance
    """
    model = WeldNet(num_classes=num_classes, in_channels=in_channels)
    if pretrained:
        raise NotImplementedError("Pretrained weights are not available yet")
    return model


def weldnet_small(num_classes=6, in_channels=3):
    """Create a smaller variant of WeldNet with width_multiplier=0.5"""
    return WeldNet(num_classes=num_classes, in_channels=in_channels, width_multiplier=0.5)


def weldnet_large(num_classes=6, in_channels=3):
    """Create a larger variant of WeldNet with width_multiplier=1.5"""
    return WeldNet(num_classes=num_classes, in_channels=in_channels, width_multiplier=1.5)


if __name__ == '__main__':
    # Test the model
    model = weldnet(num_classes=6, in_channels=3)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
