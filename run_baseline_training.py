#!/usr/bin/env python3
"""
Runner script for vanilla baseline experiments.

This script trains simple ResNet50 models with different embedding dimensions
as baselines to compare against MWRL.
"""

import subprocess
import sys
import os
from datetime import datetime


def run_vanilla_baseline(embedding_dim, exp_suffix=""):
    """
    Run vanilla baseline training with specified embedding dimension.
    
    Args:
        embedding_dim: Embedding dimension (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
        exp_suffix: Optional suffix for experiment name
    """
    # Experiment name
    exp_name = f"vanilla_dim_dim_{embedding_dim}{exp_suffix}"
    
    # Create directories
    os.makedirs(f"runs/{exp_name}", exist_ok=True)
    os.makedirs(f"checkpoints/{exp_name}", exist_ok=True)
    os.makedirs(f"results/{exp_name}", exist_ok=True)
    
    cmd = [
        sys.executable, "train_baseline.py",
        "--batch-size", "128",
        "--epochs", "100",
        "--lr", "0.1",
        "--embedding-dim", str(embedding_dim),
        "--num-classes", "100",
        "--label-smoothing", "0.1",
        "--eval-interval", "5",
        "--num-gpus", "4",
        "--log-dir", f"runs/{exp_name}",
        "--checkpoint-dir", f"checkpoints/{exp_name}",
        "--csv-dir", f"results/{exp_name}",
        "--grad-clip", "1.0",
        "--pretrained"
    ]
    
    print("="*70)
    print(f"Vanilla Baseline Training - {embedding_dim}D Embeddings")
    print("="*70)
    print(f"Experiment: {exp_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nCommand:")
    print(" ".join(cmd))
    print("\n" + "="*70)
    print("Outputs will be saved to:")
    print(f"  - TensorBoard logs:  runs/{exp_name}/")
    print(f"  - Checkpoints:       checkpoints/{exp_name}/")
    print(f"  - CSV results:       results/{exp_name}/")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print(f"✓ Training completed successfully for {embedding_dim}D!")
        print("="*70)
        print(f"\nResults:")
        print(f"  - Best model:     checkpoints/{exp_name}/best_model.pt")
        print(f"  - Metrics CSV:    results/{exp_name}/validation_metrics.csv")
        print(f"  - Loss CSV:       results/{exp_name}/training_loss.csv")
        print(f"  - Config JSON:    results/{exp_name}/training_config.json")
        print("="*70 + "\n")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
        return 1


def run_all_baselines():
    """Run baseline experiments for multiple embedding dimensions."""
    
    # Dimensions to test (choose subset for faster experimentation)
    dimensions = [2048]  # Low, low-medium, and full dimension
    # Full set: [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    print("\n" + "="*70)
    print("RUNNING VANILLA BASELINE EXPERIMENTS")
    print("="*70)
    print(f"Total experiments: {len(dimensions)}")
    print(f"Dimensions to test: {dimensions}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    results = {}
    
    for i, dim in enumerate(dimensions, 1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i}/{len(dimensions)}: {dim}-dimensional embeddings")
        print(f"{'='*70}\n")
        
        exit_code = run_vanilla_baseline(dim)
        results[dim] = exit_code
        
        if exit_code != 0:
            print(f"\n⚠ Warning: Experiment for {dim}D failed!")
            user_input = input("Continue with remaining experiments? (y/n): ")
            if user_input.lower() != 'y':
                break
    
    # Print summary
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults summary:")
    for dim, code in results.items():
        status = "✓ SUCCESS" if code == 0 else "✗ FAILED"
        print(f"  {dim}D: {status}")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Compare results across dimensions:")
    print("     python compare_baselines.py")
    print("\n  2. View individual results:")
    print("     tensorboard --logdir runs/")
    print("\n  3. Analyze CSV data:")
    print("     python analyze_csv_results.py --csv-dir results/vanilla_dim8")
    print("="*70 + "\n")
    
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run vanilla baseline experiments')
    parser.add_argument('--dim', type=int, default=None,
                        help='Run single experiment with specified dimension')
    parser.add_argument('--all', action='store_true',
                        help='Run all baseline experiments')
    
    args = parser.parse_args()
    
    if args.dim:
        # Run single experiment
        exit_code = run_vanilla_baseline(args.dim)
    elif args.all:
        # Run all experiments
        exit_code = run_all_baselines()
    else:
        # Default: run key dimensions
        print("Running baseline experiments for key dimensions: 8D, 16D, 2048D")
        print("Use --all to run all dimensions, or --dim N for a specific dimension\n")
        exit_code = run_all_baselines()
    
    sys.exit(exit_code)