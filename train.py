"""
Training script for WeldNet

This script trains the WeldNet model on welding defect datasets.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from nets import weldnet, weldnet_small, weldnet_large
from utils import (
    create_dataloaders, train_one_epoch, evaluate,
    plot_confusion_matrix, plot_training_history,
    save_checkpoint, load_checkpoint, EarlyStopping
)


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


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, num_classes, class_names = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"\nCreating model: {args.model_type}")
    model = get_model(args.model_type, num_classes, args.in_channels)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _, _, _ = load_checkpoint(model, None, args.resume, device)
        start_epoch += 1
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, restore_best_weights=True) if args.early_stopping else None
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device, class_names if epoch == args.epochs - 1 else None
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, best_model_path)
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, checkpoint_path)
        
        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, final_model_path)
    
    # Plot training history
    history_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, train_accs, val_accs, history_path)
    print(f"\nTraining history saved to {history_path}")
    
    # Plot confusion matrix for final validation
    print("\nGenerating final confusion matrix...")
    _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, device, class_names)
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(final_labels, final_preds, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WeldNet for welding defect classification')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True,
                       help='Path to validation data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs (default: outputs)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='weldnet',
                       choices=['weldnet', 'weldnet_small', 'weldnet_large'],
                       help='Model architecture to use (default: weldnet)')
    parser.add_argument('--in_channels', type=int, default=3,
                       help='Number of input channels (default: 3 for RGB)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer to use (default: adam)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'none'],
                       help='Learning rate scheduler (default: cosine)')
    
    # Data parameters
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    # Other parameters
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA training')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=15,
                       help='Patience for early stopping (default: 15)')
    
    args = parser.parse_args()
    main(args)
