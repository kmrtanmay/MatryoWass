import os
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import csv
import json
from datetime import datetime

# Import the improved MWRL model
from models.mwrl_new import create_mwrl_model
from data import ImageNet100Dataset, load_imagenet100
from utils import (
    print_gpu_info,
    setup_mig_environment,
    setup,
    cleanup,
    get_device,
    wrap_ddp_model,
    create_distributed_sampler
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MWRL on ImageNet-100 with Distributed Data Parallel')
    
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
    parser.add_argument('--nesting-list', type=str, default='8,16,32,64,128,256,512,1024,2048',
                        help='comma-separated list of nesting dimensions')
    parser.add_argument('--ce-weights', type=str, default=None,
                        help='comma-separated list of CE weights for each dimension')
    
    # MWRL specific parameters
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='weight for MRL (CE) loss (default: 1.0, set to 0 for Wasserstein-only training)')
    parser.add_argument('--wasserstein-weight', type=float, default=0.1,
                        help='weight for Wasserstein regularization (default: 0.1)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta parameter for Wasserstein cost function (default: 1.0)')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='entropic regularization for Sinkhorn (default: 0.5, larger = more stable)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing value (default: 0.1)')
    
    # Training settings
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluation interval in epochs (default: 5)')
    parser.add_argument('--multidim-plot-interval', type=int, default=20,
                        help='interval for multi-dimensional performance plots (default: 20)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers per GPU (default: 4)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='gradient clipping value (default: 1.0)')
    
    # Infrastructure
    parser.add_argument('--log-dir', type=str, default='runs/mwrl_distributed',
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/mwrl',
                        help='checkpoint directory')
    parser.add_argument('--csv-dir', type=str, default='results/mwrl',
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


def compute_metrics(logits, targets, k_values=[1, 5]):
    """
    Compute Top-K accuracies.
    
    Args:
        logits: (B, num_classes) logits
        targets: (B,) ground truth labels
        k_values: list of K values for Top-K accuracy
        
    Returns:
        Dictionary with Top-K accuracies
    """
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
    """
    Compute mean Average Precision at K (mAP@K).
    
    Args:
        logits: (B, num_classes) logits
        targets: (B,) ground truth labels
        k: number of top predictions to consider
        
    Returns:
        mAP@K value
    """
    batch_size = targets.size(0)
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    
    ap_sum = 0.0
    for i in range(batch_size):
        target = targets[i].item()
        predictions = pred[i].tolist()
        
        # Check if target is in top-k predictions
        if target in predictions:
            # Position of correct prediction (1-indexed)
            position = predictions.index(target) + 1
            # Average Precision for single query with one relevant item
            ap = 1.0 / position
        else:
            ap = 0.0
        
        ap_sum += ap
    
    return ap_sum / batch_size


def validate(model, val_loader, device, epoch, writer=None):
    """Validation function that computes accuracies for each dimension."""
    model.eval()
    nesting_list = model.nesting_list
    
    # Track metrics for each dimension
    metrics_per_dim = {
        dim: {'top1': 0, 'top5': 0, 'map10': 0.0, 'count': 0} 
        for dim in nesting_list
    }
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            targets = targets.to(device)
            
            # Classify at each dimension
            for i, dim in enumerate(nesting_list):
                logits = model.classify_at_dimension(images, dim)
                
                # Compute Top-1 and Top-5 accuracy
                metrics, batch_size = compute_metrics(logits, targets, k_values=[1, 5])
                metrics_per_dim[dim]['top1'] += metrics['top1']
                metrics_per_dim[dim]['top5'] += metrics['top5']
                
                # Compute mAP@10
                map10 = compute_map_at_k(logits, targets, k=10)
                metrics_per_dim[dim]['map10'] += map10 * batch_size
                metrics_per_dim[dim]['count'] += batch_size
    
    # Compute final metrics
    results = {}
    for dim in nesting_list:
        count = metrics_per_dim[dim]['count']
        results[dim] = {
            'top1': 100.0 * metrics_per_dim[dim]['top1'] / count,
            'top5': 100.0 * metrics_per_dim[dim]['top5'] / count,
            'map10': 100.0 * metrics_per_dim[dim]['map10'] / count
        }
        
        # Log to tensorboard
        if writer:
            writer.add_scalar(f'Accuracy_Top1/dim_{dim}', results[dim]['top1'], epoch)
            writer.add_scalar(f'Accuracy_Top5/dim_{dim}', results[dim]['top5'], epoch)
            writer.add_scalar(f'mAP@10/dim_{dim}', results[dim]['map10'], epoch)
    
    # Compute averages across all dimensions
    avg_top1 = np.mean([results[dim]['top1'] for dim in nesting_list])
    avg_top5 = np.mean([results[dim]['top5'] for dim in nesting_list])
    avg_map10 = np.mean([results[dim]['map10'] for dim in nesting_list])
    
    if writer:
        writer.add_scalar('Accuracy_Top1/average', avg_top1, epoch)
        writer.add_scalar('Accuracy_Top5/average', avg_top5, epoch)
        writer.add_scalar('mAP@10/average', avg_map10, epoch)
    
    return results, {'top1': avg_top1, 'top5': avg_top5, 'map10': avg_map10}


def plot_multidim_performance(results, nesting_list, epoch, save_path):
    """
    Create a multi-line plot showing performance across all embedding dimensions.
    
    Args:
        results: Dictionary with metrics for each dimension
        nesting_list: List of embedding dimensions
        epoch: Current epoch number
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    dimensions = sorted(nesting_list)
    top1_scores = [results[dim]['top1'] for dim in dimensions]
    top5_scores = [results[dim]['top5'] for dim in dimensions]
    map10_scores = [results[dim]['map10'] for dim in dimensions]
    
    # Plot Top-1 Accuracy
    axes[0].plot(dimensions, top1_scores, marker='o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Embedding Dimension', fontsize=12)
    axes[0].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    axes[0].set_title(f'Top-1 Accuracy vs Dimension (Epoch {epoch})', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log', base=2)
    
    # Plot Top-5 Accuracy
    axes[1].plot(dimensions, top5_scores, marker='s', linewidth=2, markersize=6, color='orange')
    axes[1].set_xlabel('Embedding Dimension', fontsize=12)
    axes[1].set_ylabel('Top-5 Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Top-5 Accuracy vs Dimension (Epoch {epoch})', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log', base=2)
    
    # Plot mAP@10
    axes[2].plot(dimensions, map10_scores, marker='^', linewidth=2, markersize=6, color='green')
    axes[2].set_xlabel('Embedding Dimension', fontsize=12)
    axes[2].set_ylabel('mAP@10 (%)', fontsize=12)
    axes[2].set_title(f'mAP@10 vs Dimension (Epoch {epoch})', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_to_csv(results, avg_metrics, epoch, csv_path, mode='a'):
    """
    Save validation metrics to CSV file for later analysis.
    
    Args:
        results: Dictionary with per-dimension metrics
        avg_metrics: Dictionary with average metrics
        epoch: Current epoch number
        csv_path: Path to CSV file
        mode: File open mode ('w' for write/overwrite, 'a' for append)
    """
    dimensions = sorted(results.keys())
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    write_header = (mode == 'w') or not file_exists
    
    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            # Write header
            header = ['epoch', 'timestamp']
            for dim in dimensions:
                header.extend([f'dim_{dim}_top1', f'dim_{dim}_top5', f'dim_{dim}_map10'])
            header.extend(['avg_top1', 'avg_top5', 'avg_map10'])
            writer.writerow(header)
        
        # Write data
        row = [epoch, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        for dim in dimensions:
            row.extend([
                f"{results[dim]['top1']:.4f}",
                f"{results[dim]['top5']:.4f}",
                f"{results[dim]['map10']:.4f}"
            ])
        row.extend([
            f"{avg_metrics['top1']:.4f}",
            f"{avg_metrics['top5']:.4f}",
            f"{avg_metrics['map10']:.4f}"
        ])
        writer.writerow(row)


def save_training_config(args, csv_dir):
    """
    Save training configuration to JSON file for reference.
    
    Args:
        args: Argument namespace
        csv_dir: Directory to save config file
    """
    config_path = os.path.join(csv_dir, 'training_config.json')
    
    config = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'alpha': args.alpha,
        'wasserstein_weight': args.wasserstein_weight,
        'beta': args.beta,
        'epsilon': args.epsilon,
        'label_smoothing': args.label_smoothing,
        'nesting_list': args.nesting_list,
        'num_gpus': args.num_gpus,
        'pretrained': args.pretrained,
        'eval_interval': args.eval_interval,
        'multidim_plot_interval': args.multidim_plot_interval
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Training configuration saved to {config_path}")


def save_loss_history_to_csv(epoch_losses, csv_path):
    """
    Save loss history to CSV file.
    
    Args:
        epoch_losses: Dictionary with loss values for the epoch
        csv_path: Path to CSV file
    """
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            # Write header
            writer.writerow([
                'epoch', 'timestamp', 'total_loss', 'ce_loss', 
                'wasserstein_loss', 'learning_rate'
            ])
        
        # Write data
        writer.writerow([
            epoch_losses['epoch'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f"{epoch_losses['total_loss']:.6f}",
            f"{epoch_losses['ce_loss']:.6f}",
            f"{epoch_losses['wasserstein_loss']:.6f}",
            f"{epoch_losses['learning_rate']:.8f}"
        ])


def save_final_summary(results, avg_metrics, best_epoch, csv_dir):
    """
    Save final training summary with best results.
    
    Args:
        results: Dictionary with per-dimension metrics
        avg_metrics: Dictionary with average metrics
        best_epoch: Epoch number of best results
        csv_dir: Directory to save summary
    """
    summary_path = os.path.join(csv_dir, 'final_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MWRL TRAINING - FINAL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Average Performance:\n")
        f.write(f"  Top-1 Accuracy:  {avg_metrics['top1']:.2f}%\n")
        f.write(f"  Top-5 Accuracy:  {avg_metrics['top5']:.2f}%\n")
        f.write(f"  mAP@10:          {avg_metrics['map10']:.2f}%\n\n")
        
        f.write("Per-Dimension Performance:\n")
        f.write(f"  {'Dim':<8} {'Top-1':<10} {'Top-5':<10} {'mAP@10':<10}\n")
        f.write(f"  {'-'*40}\n")
        
        for dim in sorted(results.keys()):
            f.write(f"  {dim:<8} {results[dim]['top1']:<10.2f} "
                   f"{results[dim]['top5']:<10.2f} {results[dim]['map10']:<10.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Final summary saved to {summary_path}")


def plot_multidim_performance(results, nesting_list, epoch, save_path):
    """
    Create a multi-line plot showing performance across all embedding dimensions.
    
    Args:
        results: Dictionary with metrics for each dimension
        nesting_list: List of embedding dimensions
        epoch: Current epoch number
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    dimensions = sorted(nesting_list)
    top1_scores = [results[dim]['top1'] for dim in dimensions]
    top5_scores = [results[dim]['top5'] for dim in dimensions]
    map10_scores = [results[dim]['map10'] for dim in dimensions]
    
    # Plot Top-1 Accuracy
    axes[0].plot(dimensions, top1_scores, marker='o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Embedding Dimension', fontsize=12)
    axes[0].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    axes[0].set_title(f'Top-1 Accuracy vs Dimension (Epoch {epoch})', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log', base=2)
    
    # Plot Top-5 Accuracy
    axes[1].plot(dimensions, top5_scores, marker='s', linewidth=2, markersize=6, color='orange')
    axes[1].set_xlabel('Embedding Dimension', fontsize=12)
    axes[1].set_ylabel('Top-5 Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Top-5 Accuracy vs Dimension (Epoch {epoch})', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log', base=2)
    
    # Plot mAP@10
    axes[2].plot(dimensions, map10_scores, marker='^', linewidth=2, markersize=6, color='green')
    axes[2].set_xlabel('Embedding Dimension', fontsize=12)
    axes[2].set_ylabel('mAP@10 (%)', fontsize=12)
    axes[2].set_title(f'mAP@10 vs Dimension (Epoch {epoch})', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_on_gpu(rank, world_size, args):
    """Training function for each GPU."""
    # Parse configurations
    nesting_list = [int(dim) for dim in args.nesting_list.split(',')]
    ce_weights = None
    if args.ce_weights:
        ce_weights = [float(w) for w in args.ce_weights.split(',')]
    
    # Set up distributed environment
    setup_mig_environment(rank, world_size)
    setup(rank, world_size)
    device = get_device(rank)
    
    # Create directories
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.csv_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'multidim_plots'), exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        
        # Save training configuration
        save_training_config(args, args.csv_dir)
        
        # Define CSV paths
        metrics_csv = os.path.join(args.csv_dir, 'validation_metrics.csv')
        loss_csv = os.path.join(args.csv_dir, 'training_loss.csv')
    else:
        writer = None
        metrics_csv = None
        loss_csv = None
    
    try:
        # Create model
        model = create_mwrl_model(
            nesting_list=nesting_list,
            num_classes=100,
            pretrained=args.pretrained,
            ce_weights=ce_weights,
            alpha=args.alpha,
            wasserstein_weight=args.wasserstein_weight,
            beta=args.beta,
            epsilon=args.epsilon,
            label_smoothing=args.label_smoothing
        )
        model = model.to(device)
        
        # Wrap with DDP (use device_id=0 for MIG instances)
        model = wrap_ddp_model(model, device_id=0)
        
        # Create optimizer
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
        best_results = None
        best_avg_metrics = None
        
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
                batch_size=args.batch_size // 4,  # Smaller batch for validation
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
        
        # Training loop
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            
            # Training metrics
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_wasserstein_loss = 0.0
            num_batches = 0
            
            # Progress bar for rank 0
            if rank == 0:
                pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Forward pass
                loss, loss_dict = model(images, targets)
                
                # Skip batch if loss is NaN
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    if rank == 0:
                        print(f"Warning: Invalid loss at batch {batch_idx}, skipping")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss_dict['total_loss']
                epoch_ce_loss += loss_dict['ce_loss']
                epoch_wasserstein_loss += loss_dict['wasserstein_loss']
                num_batches += 1
                
                # Update progress bar and log
                if rank == 0:
                    pbar.update(1)
                    if batch_idx % 10 == 0 and writer:
                        global_step = epoch * len(train_loader) + batch_idx
                        writer.add_scalar('Loss/batch_total', loss_dict['total_loss'], global_step)
                        writer.add_scalar('Loss/batch_ce', loss_dict['ce_loss'], global_step)
                        writer.add_scalar('Loss/batch_wasserstein', 
                                        loss_dict['wasserstein_loss'], global_step)
            
            if rank == 0:
                pbar.close()
            
            # Update learning rate
            scheduler.step()
            
            # Log epoch metrics
            if rank == 0 and num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_ce_loss = epoch_ce_loss / num_batches
                avg_wasserstein_loss = epoch_wasserstein_loss / num_batches
                
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Average Loss: {avg_loss:.4f}")
                print(f"  Average CE Loss: {avg_ce_loss:.4f}")
                print(f"  Average Wasserstein Loss: {avg_wasserstein_loss:.4f}")
                print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
                
                if writer:
                    writer.add_scalar('Loss/epoch_total', avg_loss, epoch)
                    writer.add_scalar('Loss/epoch_ce', avg_ce_loss, epoch)
                    writer.add_scalar('Loss/epoch_wasserstein', avg_wasserstein_loss, epoch)
                    writer.add_scalar('LR/epoch', scheduler.get_last_lr()[0], epoch)
                
                # Save loss history to CSV
                epoch_loss_data = {
                    'epoch': epoch,
                    'total_loss': avg_loss,
                    'ce_loss': avg_ce_loss,
                    'wasserstein_loss': avg_wasserstein_loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                save_loss_history_to_csv(epoch_loss_data, loss_csv)
            
            # Validation
            if rank == 0 and (epoch + 1) % args.eval_interval == 0:
                results, avg_metrics = validate(model.module, val_loader, device, epoch, writer)
                
                print(f"\nValidation Results (Epoch {epoch}):")
                print(f"  {'Dim':<8} {'Top-1':<10} {'Top-5':<10} {'mAP@10':<10}")
                print(f"  {'-'*40}")
                for dim in sorted(results.keys()):
                    print(f"  {dim:<8} {results[dim]['top1']:<10.2f} "
                          f"{results[dim]['top5']:<10.2f} {results[dim]['map10']:<10.2f}")
                print(f"  {'-'*40}")
                print(f"  {'Average':<8} {avg_metrics['top1']:<10.2f} "
                      f"{avg_metrics['top5']:<10.2f} {avg_metrics['map10']:<10.2f}")
                
                # Save metrics to CSV
                save_metrics_to_csv(results, avg_metrics, epoch, metrics_csv, mode='a')
                
                # Save best model based on Top-1 accuracy
                if avg_metrics['top1'] > best_top1:
                    best_top1 = avg_metrics['top1']
                    best_epoch = epoch
                    best_results = results.copy()
                    best_avg_metrics = avg_metrics.copy()
                    
                    print(f"\n✓ New best model! Top-1 accuracy: {best_top1:.2f}%")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_top1': best_top1,
                        'results': results,
                        'avg_metrics': avg_metrics,
                        'args': args
                    }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
                
                # Create multi-dimensional performance plot
                if (epoch + 1) % args.multidim_plot_interval == 0:
                    plot_path = os.path.join(args.log_dir, 'multidim_plots', 
                                            f'performance_epoch_{epoch}.png')
                    plot_multidim_performance(results, nesting_list, epoch, plot_path)
                    
                    # Log plot to tensorboard
                    if writer:
                        img = plt.imread(plot_path)
                        writer.add_image('Performance/multi_dimensional', 
                                       img.transpose(2, 0, 1), epoch)
            
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
            
            # Save final summary with best results
            if best_results is not None:
                save_final_summary(best_results, best_avg_metrics, best_epoch, args.csv_dir)
            
            print(f"\n✓ Training completed! Best Top-1 accuracy: {best_top1:.2f}% (Epoch {best_epoch})")
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
    
    # Check if GeomLoss or POT is installed
    geomloss_available = False
    pot_available = False
    
    try:
        import geomloss
        geomloss_available = True
        print("✓ GeomLoss found - using fully differentiable optimal transport (recommended)")
    except ImportError:
        pass
    
    if not geomloss_available:
        try:
            import ot
            pot_available = True
            print("✓ POT found - using POT fallback")
            print("  Note: For better training results, consider installing GeomLoss:")
            print("  pip install geomloss")
        except ImportError:
            pass
    
    if not geomloss_available and not pot_available:
        print("✗ Error: Either GeomLoss or POT is required for MWRL training.")
        print("\nRecommended (fully differentiable):")
        print("  pip install geomloss")
        print("\nAlternative (fallback):")
        print("  pip install pot")
        return
    
    # Print configuration
    print("\n" + "="*60)
    print("MWRL Distributed Training Configuration")
    print("="*60)
    print(f"  Batch size (global):      {args.batch_size}")
    print(f"  Batch size (per GPU):     {args.batch_size // args.num_gpus}")
    print(f"  Epochs:                   {args.epochs}")
    print(f"  Learning rate:            {args.lr}")
    print(f"  Momentum:                 {args.momentum}")
    print(f"  Weight decay:             {args.weight_decay}")
    print(f"  Gradient clipping:        {args.grad_clip}")
    print(f"\nMWRL Parameters:")
    print(f"  Alpha (CE weight):        {args.alpha} {'(Wasserstein-only mode)' if args.alpha == 0 else ''}")
    print(f"  Wasserstein weight (λ):   {args.wasserstein_weight}")
    print(f"  Beta (remainder penalty): {args.beta}")
    print(f"  Epsilon (Sinkhorn blur):  {args.epsilon}")
    print(f"  Label smoothing:          {args.label_smoothing}")
    print(f"  Nesting dimensions:       {args.nesting_list}")
    print(f"\nMetrics:")
    print(f"  - Top-1 Accuracy")
    print(f"  - Top-5 Accuracy")
    print(f"  - mAP@10")
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
    print(f"  Multi-dim plot interval:  Every {args.multidim_plot_interval} epochs")
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