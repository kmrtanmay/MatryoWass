#!/usr/bin/env python3
"""
Multi-Experiment Analysis and Plotting Tool

This script reads results from multiple MWRL/baseline experiments and creates
comprehensive comparison plots for papers.

Usage:
    python plot_multi_experiments.py \
        --experiment-dirs results/exp1 results/exp2 results/exp3 \
        --labels "Alpha=1.0" "Alpha=0.5" "Alpha=0.0" \
        --output-dir paper_plots
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import json
from pathlib import Path
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def parse_args():
    parser = argparse.ArgumentParser(description='Compare multiple MWRL experiments')
    
    parser.add_argument('--experiment-dirs', type=str, nargs='+', required=True,
                        help='Directories containing experiment results (e.g., results/exp1 results/exp2)')
    parser.add_argument('--labels', type=str, nargs='+', required=True,
                        help='Labels for each experiment (must match number of dirs)')
    parser.add_argument('--output-dir', type=str, default='multi_experiment_plots',
                        help='Output directory for plots')
    parser.add_argument('--dimensions', type=str, default=None,
                        help='Comma-separated dimensions to plot (default: all)')
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=['top1', 'top5', 'map10'],
                        help='Metrics to plot')
    parser.add_argument('--figsize', type=str, default='10,6',
                        help='Figure size as width,height (default: 10,6)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures (default: 300)')
    
    return parser.parse_args()


def load_experiment(exp_dir):
    """
    Load experiment data from directory.
    
    Returns:
        dict with keys: 'config', 'metrics', 'loss', 'name'
    """
    try:
        config_path = os.path.join(exp_dir, 'training_config.json')
        metrics_path = os.path.join(exp_dir, 'validation_metrics.csv')
        loss_path = os.path.join(exp_dir, 'training_loss.csv')
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load metrics and loss
        metrics_df = pd.read_csv(metrics_path)
        loss_df = pd.read_csv(loss_path)
        
        # Determine experiment type
        exp_type = config.get('model_type', 'mwrl')
        
        return {
            'config': config,
            'metrics': metrics_df,
            'loss': loss_df,
            'name': os.path.basename(exp_dir),
            'type': exp_type
        }
    except Exception as e:
        print(f"Warning: Could not load {exp_dir}: {e}")
        return None


def plot_training_loss_comparison(experiments, labels, output_dir, figsize):
    """
    Plot training loss curves for all experiments.
    """
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*1.5, figsize[1]))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for idx, (exp, label) in enumerate(zip(experiments, labels)):
        loss_df = exp['loss']
        epochs = loss_df['epoch'].values
        
        # Total loss
        if 'total_loss' in loss_df.columns:
            total_loss = loss_df['total_loss'].values
            axes[0].plot(epochs, total_loss, linewidth=2.5, color=colors[idx], 
                        label=label, alpha=0.85)
        
        # CE loss
        if 'ce_loss' in loss_df.columns:
            ce_loss = loss_df['ce_loss'].values
            axes[1].plot(epochs, ce_loss, linewidth=2.5, color=colors[idx], 
                        label=label, alpha=0.85)
        elif 'loss' in loss_df.columns:
            # Vanilla baseline
            loss = loss_df['loss'].values
            axes[1].plot(epochs, loss, linewidth=2.5, color=colors[idx], 
                        label=label, alpha=0.85, linestyle='--')
        
        # Wasserstein loss
        if 'wasserstein_loss' in loss_df.columns:
            w_loss = loss_df['wasserstein_loss'].values
            axes[2].plot(epochs, w_loss, linewidth=2.5, color=colors[idx], 
                        label=label, alpha=0.85)
    
    # Format axes
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Total Loss', fontsize=13, fontweight='bold')
    axes[0].set_title('Total Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10, loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='bold')
    axes[1].set_title('CE Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Wasserstein Loss', fontsize=13, fontweight='bold')
    axes[2].set_title('Wasserstein Regularization', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10, loc='best', framealpha=0.9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_comparison.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✓ Loss comparison saved to {output_path}")


def plot_validation_metrics_comparison(experiments, labels, output_dir, figsize, metrics):
    """
    Plot validation metrics evolution for all experiments.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0]*n_metrics/2, figsize[1]))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    metric_names = {
        'top1': 'Top-1 Accuracy',
        'top5': 'Top-5 Accuracy',
        'map10': 'mAP@10'
    }
    
    for metric_idx, metric in enumerate(metrics):
        for exp_idx, (exp, label) in enumerate(zip(experiments, labels)):
            metrics_df = exp['metrics']
            epochs = metrics_df['epoch'].values
            
            # Check for average or direct metric
            if f'avg_{metric}' in metrics_df.columns:
                values = metrics_df[f'avg_{metric}'].values
            elif metric in metrics_df.columns:
                values = metrics_df[metric].values
            else:
                continue
            
            axes[metric_idx].plot(epochs, values, linewidth=2.5, 
                                 marker='o', markersize=4,
                                 color=colors[exp_idx], label=label, alpha=0.85)
        
        axes[metric_idx].set_xlabel('Epoch', fontsize=13, fontweight='bold')
        axes[metric_idx].set_ylabel(f'{metric_names.get(metric, metric)} (%)', 
                                    fontsize=13, fontweight='bold')
        axes[metric_idx].set_title(metric_names.get(metric, metric), 
                                   fontsize=14, fontweight='bold')
        axes[metric_idx].legend(fontsize=10, loc='best', framealpha=0.9)
        axes[metric_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics comparison saved to {output_path}")


def plot_dimension_performance(experiments, labels, output_dir, figsize, dimensions=None):
    """
    Plot performance across dimensions for MWRL experiments.
    """
    # Filter MWRL experiments
    mwrl_exps = [(exp, label) for exp, label in zip(experiments, labels) 
                 if exp['type'] != 'vanilla_baseline']
    
    if not mwrl_exps:
        print("No MWRL experiments found for dimension plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*1.5, figsize[1]))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(mwrl_exps)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for exp_idx, (exp, label) in enumerate(mwrl_exps):
        # Get dimensions from config
        nesting_list_str = exp['config'].get('nesting_list', '8,16,32,64,128,256,512,1024,2048')
        all_dims = [int(d) for d in nesting_list_str.split(',')]
        
        # Filter dimensions if specified
        if dimensions:
            dims_to_plot = [d for d in all_dims if d in dimensions]
        else:
            dims_to_plot = all_dims
        
        # Get last epoch metrics
        last_metrics = exp['metrics'].iloc[-1]
        
        # Extract per-dimension metrics
        top1_scores = [last_metrics[f'dim_{dim}_top1'] for dim in dims_to_plot]
        top5_scores = [last_metrics[f'dim_{dim}_top5'] for dim in dims_to_plot]
        map10_scores = [last_metrics[f'dim_{dim}_map10'] for dim in dims_to_plot]
        
        # Plot
        marker = markers[exp_idx % len(markers)]
        
        axes[0].plot(dims_to_plot, top1_scores, 
                    marker=marker, linewidth=2.5, markersize=8,
                    color=colors[exp_idx], label=label, alpha=0.85)
        
        axes[1].plot(dims_to_plot, top5_scores,
                    marker=marker, linewidth=2.5, markersize=8,
                    color=colors[exp_idx], label=label, alpha=0.85)
        
        axes[2].plot(dims_to_plot, map10_scores,
                    marker=marker, linewidth=2.5, markersize=8,
                    color=colors[exp_idx], label=label, alpha=0.85)
    
    # Format axes
    for ax, title in zip(axes, ['Top-1 Accuracy', 'Top-5 Accuracy', 'mAP@10']):
        ax.set_xlabel('Embedding Dimension', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{title} (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{title} vs Dimension', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dimension_performance.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✓ Dimension performance saved to {output_path}")


def plot_top1_only_dimension(experiments, labels, output_dir, figsize, dimensions=None):
    """
    Plot ONLY Top-1 accuracy across dimensions (larger, clearer plot).
    Perfect for highlighting dimension scaling in papers.
    """
    # Filter MWRL experiments
    mwrl_exps = [(exp, label) for exp, label in zip(experiments, labels) 
                 if exp['type'] != 'vanilla_baseline']
    
    if not mwrl_exps:
        print("No MWRL experiments found for Top-1 dimension plot")
        return
    
    # Create larger single plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(mwrl_exps)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for exp_idx, (exp, label) in enumerate(mwrl_exps):
        # Get dimensions from config
        nesting_list_str = exp['config'].get('nesting_list', '8,16,32,64,128,256,512,1024,2048')
        all_dims = [int(d) for d in nesting_list_str.split(',')]
        
        # Filter dimensions if specified
        if dimensions:
            dims_to_plot = [d for d in all_dims if d in dimensions]
        else:
            dims_to_plot = all_dims
        
        # Get last epoch metrics
        last_metrics = exp['metrics'].iloc[-1]
        
        # Extract Top-1 scores
        top1_scores = [last_metrics[f'dim_{dim}_top1'] for dim in dims_to_plot]
        
        # Plot with thicker lines and larger markers
        marker = markers[exp_idx % len(markers)]
        ax.plot(dims_to_plot, top1_scores, 
                marker=marker, linewidth=3, markersize=10,
                color=colors[exp_idx], label=label, alpha=0.85)
    
    # Format axes
    ax.set_xlabel('Embedding Dimension', fontsize=15, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=15, fontweight='bold')
    ax.set_title('Top-1 Accuracy Across Dimensions', fontsize=16, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.95, 
             edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # Add minor gridlines for better readability
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'top1_across_dimensions.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✓ Top-1 dimension plot saved to {output_path}")


def plot_final_results_bars(experiments, labels, output_dir, figsize):
    """
    Create grouped bar chart of final results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*1.5, figsize[1]))
    
    metrics = ['top1', 'top5', 'map10']
    metric_names = ['Top-1 Accuracy', 'Top-5 Accuracy', 'mAP@10']
    
    x = np.arange(len(labels))
    width = 0.7
    
    for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = []
        
        for exp in experiments:
            metrics_df = exp['metrics']
            
            # Get best value
            if f'avg_{metric}' in metrics_df.columns:
                best_val = metrics_df[f'avg_{metric}'].max()
            elif metric in metrics_df.columns:
                best_val = metrics_df[metric].max()
            else:
                best_val = 0
            
            values.append(best_val)
        
        # Create bars
        bars = axes[metric_idx].bar(x, values, width, alpha=0.8)
        
        # Color bars
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        axes[metric_idx].set_xlabel('Experiment', fontsize=13, fontweight='bold')
        axes[metric_idx].set_ylabel(f'{metric_name} (%)', fontsize=13, fontweight='bold')
        axes[metric_idx].set_title(f'Best {metric_name}', fontsize=14, fontweight='bold')
        axes[metric_idx].set_xticks(x)
        axes[metric_idx].set_xticklabels(labels, rotation=45, ha='right')
        axes[metric_idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[metric_idx].text(bar.get_x() + bar.get_width()/2., height,
                                 f'{val:.2f}%',
                                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'final_results_bars.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✓ Final results bars saved to {output_path}")


def plot_specific_dimension_comparison(experiments, labels, output_dir, figsize, dimension):
    """
    Compare performance at a specific dimension across experiments.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for exp_idx, (exp, label) in enumerate(zip(experiments, labels)):
        metrics_df = exp['metrics']
        epochs = metrics_df['epoch'].values
        
        # Check if this dimension exists in metrics
        col_name = f'dim_{dimension}_top1'
        if col_name in metrics_df.columns:
            values = metrics_df[col_name].values
            ax.plot(epochs, values, linewidth=2.5, marker='o', markersize=4,
                   color=colors[exp_idx], label=label, alpha=0.85)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Top-1 Accuracy @ {dimension}D', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'dim{dimension}_comparison.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✓ Dimension {dimension} comparison saved to {output_path}")


def create_results_table(experiments, labels, output_dir):
    """
    Create comprehensive results table.
    """
    results = []
    
    for exp, label in zip(experiments, labels):
        config = exp['config']
        metrics_df = exp['metrics']
        
        # Get best metrics
        if 'avg_top1' in metrics_df.columns:
            best_top1 = metrics_df['avg_top1'].max()
            best_top5 = metrics_df['avg_top5'].max()
            best_map10 = metrics_df['avg_map10'].max()
        else:
            best_top1 = metrics_df['top1'].max()
            best_top5 = metrics_df['top5'].max()
            best_map10 = metrics_df['map10'].max()
        
        # Get final metrics
        final_top1 = metrics_df.iloc[-1].get('avg_top1', metrics_df.iloc[-1].get('top1', 0))
        final_top5 = metrics_df.iloc[-1].get('avg_top5', metrics_df.iloc[-1].get('top5', 0))
        final_map10 = metrics_df.iloc[-1].get('avg_map10', metrics_df.iloc[-1].get('map10', 0))
        
        results.append({
            'Experiment': label,
            'Alpha': config.get('alpha', 'N/A'),
            'Lambda': config.get('wasserstein_weight', 'N/A'),
            'Beta': config.get('beta', 'N/A'),
            'Epsilon': config.get('epsilon', 'N/A'),
            'Best_Top1': f"{best_top1:.2f}",
            'Best_Top5': f"{best_top5:.2f}",
            'Best_mAP10': f"{best_map10:.2f}",
            'Final_Top1': f"{final_top1:.2f}",
            'Final_Top5': f"{final_top5:.2f}",
            'Final_mAP10': f"{final_map10:.2f}"
        })
    
    df = pd.DataFrame(results)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Results table saved to {csv_path}")
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'results_summary.tex')
    with open(latex_path, 'w') as f:
        f.write("% Multi-Experiment Results Table\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of MWRL Hyperparameter Configurations}\n")
        f.write("\\label{tab:mwrl_hyperparams}\n")
        f.write("\\begin{tabular}{lcccc|ccc|ccc}\n")
        f.write("\\hline\n")
        f.write("\\multirow{2}{*}{Experiment} & \\multicolumn{4}{c|}{Hyperparameters} & "
               "\\multicolumn{3}{c|}{Best} & \\multicolumn{3}{c}{Final} \\\\\n")
        f.write("& $\\alpha$ & $\\lambda$ & $\\beta$ & $\\epsilon$ & "
               "Top-1 & Top-5 & mAP@10 & Top-1 & Top-5 & mAP@10 \\\\\n")
        f.write("\\hline\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['Experiment']} & {row['Alpha']} & {row['Lambda']} & "
                   f"{row['Beta']} & {row['Epsilon']} & "
                   f"{row['Best_Top1']} & {row['Best_Top5']} & {row['Best_mAP10']} & "
                   f"{row['Final_Top1']} & {row['Final_Top5']} & {row['Final_mAP10']} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    
    print(f"✓ LaTeX table saved to {latex_path}")
    
    # Print to console
    print("\n" + "="*120)
    print("RESULTS SUMMARY")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120 + "\n")


def plot_learning_curves_grid(experiments, labels, output_dir, figsize):
    """
    Create grid of learning curves for each experiment.
    """
    n_exps = len(experiments)
    n_cols = min(3, n_exps)
    n_rows = (n_exps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows/2))
    axes = np.atleast_2d(axes).flatten()
    
    for idx, (exp, label) in enumerate(zip(experiments, labels)):
        ax = axes[idx]
        
        # Plot training loss and validation accuracy on same plot (dual y-axis)
        loss_df = exp['loss']
        metrics_df = exp['metrics']
        
        color_loss = 'tab:red'
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Training Loss', fontsize=11, fontweight='bold', color=color_loss)
        
        loss_col = 'total_loss' if 'total_loss' in loss_df.columns else 'loss'
        ax.plot(loss_df['epoch'], loss_df[loss_col], 
               color=color_loss, linewidth=2, alpha=0.8, label='Loss')
        ax.tick_params(axis='y', labelcolor=color_loss)
        
        # Create second y-axis for accuracy
        ax2 = ax.twinx()
        color_acc = 'tab:blue'
        ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=11, fontweight='bold', color=color_acc)
        
        metric_col = 'avg_top1' if 'avg_top1' in metrics_df.columns else 'top1'
        ax2.plot(metrics_df['epoch'], metrics_df[metric_col],
                color=color_acc, linewidth=2, alpha=0.8, marker='o', markersize=3, label='Top-1')
        ax2.tick_params(axis='y', labelcolor=color_acc)
        
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_exps, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'learning_curves_grid.png')
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✓ Learning curves grid saved to {output_path}")


def main():
    global args
    args = parse_args()
    
    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # Parse dimensions if specified
    dimensions = None
    if args.dimensions:
        dimensions = [int(d) for d in args.dimensions.split(',')]
    
    # Validate inputs
    if len(args.experiment_dirs) != len(args.labels):
        print("Error: Number of experiment directories must match number of labels")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiments
    print("\n" + "="*70)
    print("LOADING EXPERIMENTS")
    print("="*70)
    
    experiments = []
    valid_labels = []
    
    for exp_dir, label in zip(args.experiment_dirs, args.labels):
        exp = load_experiment(exp_dir)
        if exp:
            experiments.append(exp)
            valid_labels.append(label)
            print(f"✓ Loaded: {label} ({exp_dir})")
        else:
            print(f"✗ Failed: {label} ({exp_dir})")
    
    if not experiments:
        print("\nError: No experiments loaded successfully")
        return 1
    
    print(f"\nTotal experiments loaded: {len(experiments)}")
    print("="*70 + "\n")
    
    # Generate plots
    print("Generating comparison plots...\n")
    
    # 1. Training loss comparison
    plot_training_loss_comparison(experiments, valid_labels, args.output_dir, figsize)
    
    # 2. Validation metrics comparison
    plot_validation_metrics_comparison(experiments, valid_labels, args.output_dir, 
                                      figsize, args.metrics)
    
    # 3. Dimension performance (for MWRL experiments)
    plot_dimension_performance(experiments, valid_labels, args.output_dir, 
                              figsize, dimensions)
    
    # 3b. Top-1 only dimension plot (NEW - larger, clearer)
    plot_top1_only_dimension(experiments, valid_labels, args.output_dir,
                            figsize, dimensions)
    
    # 4. Final results bars
    plot_final_results_bars(experiments, valid_labels, args.output_dir, figsize)
    
    # 5. Learning curves grid
    plot_learning_curves_grid(experiments, valid_labels, args.output_dir, figsize)
    
    # 6. Specific dimension comparisons (for common dimensions)
    common_dims = [8, 16, 64, 256, 1024, 2048]
    for dim in common_dims:
        # Check if any experiment has this dimension
        has_dim = any(f'dim_{dim}_top1' in exp['metrics'].columns for exp in experiments)
        if has_dim:
            plot_specific_dimension_comparison(experiments, valid_labels, 
                                              args.output_dir, figsize, dim)
    
    # 7. Results table
    create_results_table(experiments, valid_labels, args.output_dir)
    
    print(f"\n✓ All plots saved to {args.output_dir}/")
    print("\nGenerated files:")
    print("  - loss_comparison.png (+ .pdf)")
    print("  - metrics_comparison.png (+ .pdf)")
    print("  - dimension_performance.png (+ .pdf)")
    print("  - final_results_bars.png (+ .pdf)")
    print("  - learning_curves_grid.png (+ .pdf)")
    print("  - dim{N}_comparison.png (+ .pdf) [for each dimension]")
    print("  - results_summary.csv")
    print("  - results_summary.tex")
    print("\nDone!")


if __name__ == '__main__':
    main()