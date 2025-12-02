"""
Vanilla Baseline Training Script for ImageNet-100

Simple ResNet50 training with configurable embedding dimension.
No Matryoshka nesting, no Wasserstein loss - just standard cross-entropy.

This serves as a baseline to compare against MWRL.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import csv
import json
from datetime import datetime

# Import data utilities (same as MWRL)
from data import ImageNet100Dataset, load_imagenet100
from utils import (
    setup_mig_environment,
    setup,
    cleanup,
    get_device,
    wrap_ddp_model,
    create_distributed_sampler
)


def parse_args():
    parser = argparse.ArgumentParser(description='Vanilla Baseline Training on ImageNet-100')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=256,
                        help='global batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    
    # Model parameters
    parser.add_argument('--embedding-dim', type=int, default=2048,
                        help='embedding dimension (8, 16, 32, 64, 128, 256, 512, 1024, 2048)')
    parser.add_argument('--num-classes', type=int, default=100,
                        help='number of output classes (default: 100)')
    
    # Training settings
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing value (default: 0.1)')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluation interval in epochs (default: 5)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers per GPU (default: 4)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='gradient clipping value (default: 1.0)')
    
    # Infrastructure
    parser.add_argument('--log-dir', type=str, default='runs/vanilla_baseline',
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/vanilla_baseline',
                        help='checkpoint directory')
    parser.add_argument('--csv-dir', type=str, default='results/vanilla_baseline',
                        help='directory to save CSV results')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='number of GPUs to use (default: 4)')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='distributed backend: gloo or nccl (default: nccl)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained ResNet50')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path to resume from')
    
    return parser.parse_args()


class VanillaResNet(nn.Module):
    """
    Simple ResNet50 with configurable embedding dimension.
    
    Architecture:
        ResNet50 backbone → Linear(2048, embedding_dim) → Linear(embedding_dim, num_classes)
    """
    
    def __init__(self, embedding_dim=2048, num_classes=100, pretrained=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Backbone: ResNet50
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)
        
        # Extract features (remove fc layer)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_dim = 2048
        
        # Embedding layer
        if embedding_dim == 2048:
            # No dimension reduction needed
            self.embedding_layer = nn.Identity()
        else:
            # Project to lower dimension
            self.embedding_layer = nn.Linear(self.resnet_dim, embedding_dim)
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W) input images
            
        Returns:
            logits: (B, num_classes)
        """
        # Extract features
        features = self.features(x)  # (B, 2048, 1, 1)
        features = torch.flatten(features, 1)  # (B, 2048)
        
        # Project to embedding space
        embeddings = self.embedding_layer(features)  # (B, embedding_dim)
        
        # Classify
        logits = self.classifier(embeddings)  # (B, num_classes)
        
        return logits
    
    def get_embeddings(self, x):
        """Extract embeddings without classification."""
        features = self.features(x)
        features = torch.flatten(features, 1)
        embeddings = self.embedding_layer(features)
        return embeddings


def compute_metrics(logits, targets, k_values=[1, 5]):
    """Compute Top-K accuracies."""
    batch_size = targets.size(0)
    _, pred = logits.topk(max(k_values), dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    metrics = {}
    for k in k_values:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        metrics[f'top{k}'] = correct_k.item()
    
    return metrics, batch_size


def compute_map_at_k(logits, targets, k=10):
    """Compute mean Average Precision at K."""
    batch_size = targets.size(0)
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    
    ap_sum = 0.0
    for i in range(batch_size):
        target = targets[i].item()
        predictions = pred[i].tolist()
        
        if target in predictions:
            position = predictions.index(target) + 1
            ap = 1.0 / position
        else:
            ap = 0.0
        
        ap_sum += ap
    
    return ap_sum / batch_size


def validate(model, val_loader, device, epoch, writer=None):
    """Validation function."""
    model.eval()
    
    total_top1 = 0
    total_top5 = 0
    total_map10 = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images)
            
            # Compute metrics
            metrics, batch_size = compute_metrics(logits, targets, k_values=[1, 5])
            total_top1 += metrics['top1']
            total_top5 += metrics['top5']
            
            map10 = compute_map_at_k(logits, targets, k=10)
            total_map10 += map10 * batch_size
            total_samples += batch_size
    
    # Compute final metrics
    top1_acc = 100.0 * total_top1 / total_samples
    top5_acc = 100.0 * total_top5 / total_samples
    map10 = 100.0 * total_map10 / total_samples
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Accuracy/top1', top1_acc, epoch)
        writer.add_scalar('Accuracy/top5', top5_acc, epoch)
        writer.add_scalar('mAP@10/score', map10, epoch)
    
    return {
        'top1': top1_acc,
        'top5': top5_acc,
        'map10': map10
    }


def save_metrics_to_csv(metrics, epoch, csv_path, mode='a'):
    """Save metrics to CSV."""
    file_exists = os.path.exists(csv_path)
    write_header = (mode == 'w') or not file_exists
    
    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(['epoch', 'timestamp', 'top1', 'top5', 'map10'])
        
        writer.writerow([
            epoch,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f"{metrics['top1']:.4f}",
            f"{metrics['top5']:.4f}",
            f"{metrics['map10']:.4f}"
        ])


def save_loss_to_csv(epoch, loss, lr, csv_path):
    """Save training loss to CSV."""
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['epoch', 'timestamp', 'loss', 'learning_rate'])
        
        writer.writerow([
            epoch,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f"{loss:.6f}",
            f"{lr:.8f}"
        ])


def save_training_config(args, csv_dir):
    """Save training configuration."""
    config_path = os.path.join(csv_dir, 'training_config.json')
    
    config = {
        'model_type': 'vanilla_baseline',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'embedding_dim': args.embedding_dim,
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        'num_gpus': args.num_gpus,
        'pretrained': args.pretrained,
        'eval_interval': args.eval_interval
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Configuration saved to {config_path}")


def save_final_summary(best_metrics, best_epoch, csv_dir):
    """Save final summary."""
    summary_path = os.path.join(csv_dir, 'final_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VANILLA BASELINE TRAINING - FINAL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Best Performance:\n")
        f.write(f"  Top-1 Accuracy:  {best_metrics['top1']:.2f}%\n")
        f.write(f"  Top-5 Accuracy:  {best_metrics['top5']:.2f}%\n")
        f.write(f"  mAP@10:          {best_metrics['map10']:.2f}%\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Final summary saved to {summary_path}")


def train_on_gpu(rank, world_size, args):
    """Training function for each GPU."""
    # Set up distributed environment
    setup_mig_environment(rank, world_size)
    setup(rank, world_size)
    device = get_device(rank)
    
    # Create directories
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.csv_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        
        save_training_config(args, args.csv_dir)
        
        metrics_csv = os.path.join(args.csv_dir, 'validation_metrics.csv')
        loss_csv = os.path.join(args.csv_dir, 'training_loss.csv')
    else:
        writer = None
        metrics_csv = None
        loss_csv = None
    
    try:
        # Create model
        model = VanillaResNet(
            embedding_dim=args.embedding_dim,
            num_classes=args.num_classes,
            pretrained=args.pretrained
        )
        model = model.to(device)
        
        # Wrap with DDP
        model = wrap_ddp_model(model, device_id=0)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        
        # Optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # Load checkpoint if resuming
        start_epoch = 0
        best_top1 = 0.0
        best_epoch = 0
        best_metrics = None
        
        if args.resume and rank == 0:
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_top1 = checkpoint.get('best_top1', 0.0)
        
        # Load dataset
        hf_dataset = load_imagenet100()
        
        # Data transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        # Create datasets
        train_dataset = ImageNet100Dataset(hf_dataset, split="train", 
                                         transform=train_transform)
        val_dataset = ImageNet100Dataset(hf_dataset, split="validation", 
                                       transform=val_transform)
        
        # Create distributed sampler and dataloader
        train_sampler = create_distributed_sampler(train_dataset, rank, world_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size // world_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Validation loader (only for rank 0)
        if rank == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size // 4,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
        
        # Training loop
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            
            epoch_loss = 0.0
            num_batches = 0
            
            if rank == 0:
                pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Forward pass
                logits = model(images)
                loss = criterion(logits, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                
                if rank == 0:
                    pbar.update(1)
                    if batch_idx % 10 == 0 and writer:
                        global_step = epoch * len(train_loader) + batch_idx
                        writer.add_scalar('Loss/batch', loss.item(), global_step)
            
            if rank == 0:
                pbar.close()
            
            # Update learning rate
            scheduler.step()
            
            # Log epoch metrics
            if rank == 0 and num_batches > 0:
                avg_loss = epoch_loss / num_batches
                
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Average Loss: {avg_loss:.4f}")
                print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
                
                if writer:
                    writer.add_scalar('Loss/epoch', avg_loss, epoch)
                    writer.add_scalar('LR/epoch', scheduler.get_last_lr()[0], epoch)
                
                save_loss_to_csv(epoch, avg_loss, scheduler.get_last_lr()[0], loss_csv)
            
            # Validation
            if rank == 0 and (epoch + 1) % args.eval_interval == 0:
                metrics = validate(model.module, val_loader, device, epoch, writer)
                
                print(f"\nValidation Results (Epoch {epoch}):")
                print(f"  Top-1 Accuracy: {metrics['top1']:.2f}%")
                print(f"  Top-5 Accuracy: {metrics['top5']:.2f}%")
                print(f"  mAP@10:         {metrics['map10']:.2f}%")
                
                save_metrics_to_csv(metrics, epoch, metrics_csv, mode='a')
                
                # Save best model
                if metrics['top1'] > best_top1:
                    best_top1 = metrics['top1']
                    best_epoch = epoch
                    best_metrics = metrics.copy()
                    
                    print(f"\n✓ New best model! Top-1: {best_top1:.2f}%")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_top1': best_top1,
                        'metrics': metrics,
                        'args': args
                    }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            
            # Regular checkpoint
            if rank == 0 and (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'args': args
                }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Save final model
        if rank == 0:
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'args': args
            }, os.path.join(args.checkpoint_dir, 'final_model.pt'))
            
            if best_metrics is not None:
                save_final_summary(best_metrics, best_epoch, args.csv_dir)
            
            print(f"\n✓ Training completed! Best Top-1: {best_top1:.2f}% (Epoch {best_epoch})")
            print(f"✓ Results saved to {args.csv_dir}/")
    
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if rank == 0 and writer:
            writer.close()
        cleanup()


def main():
    args = parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("Vanilla Baseline Training Configuration")
    print("="*60)
    print(f"  Embedding dimension:      {args.embedding_dim}")
    print(f"  Number of classes:        {args.num_classes}")
    print(f"  Batch size (global):      {args.batch_size}")
    print(f"  Batch size (per GPU):     {args.batch_size // args.num_gpus}")
    print(f"  Epochs:                   {args.epochs}")
    print(f"  Learning rate:            {args.lr}")
    print(f"  Momentum:                 {args.momentum}")
    print(f"  Weight decay:             {args.weight_decay}")
    print(f"  Label smoothing:          {args.label_smoothing}")
    print(f"  Gradient clipping:        {args.grad_clip}")
    print(f"\nInfrastructure:")
    print(f"  Number of GPUs:           {args.num_gpus}")
    print(f"  Backend:                  {args.backend}")
    print(f"  Workers per GPU:          {args.num_workers}")
    print(f"  Pretrained backbone:      {args.pretrained}")
    print(f"\nLogging:")
    print(f"  Log directory:            {args.log_dir}")
    print(f"  Checkpoint directory:     {args.checkpoint_dir}")
    print(f"  CSV results directory:    {args.csv_dir}")
    print(f"  Evaluation interval:      Every {args.eval_interval} epochs")
    print("="*60 + "\n")
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Launch distributed training
    print(f"Launching {args.num_gpus} training processes...\n")
    mp.spawn(
        train_on_gpu,
        args=(args.num_gpus, args),
        nprocs=args.num_gpus,
        join=True
    )


if __name__ == '__main__':
    main()