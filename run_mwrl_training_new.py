#!/usr/bin/env python3
"""
Example script to run MWRL training with CSV logging
"""

import subprocess
import sys
import os
from datetime import datetime

def run_mwrl_training():
    """Run MWRL training with example parameters"""
    
    # Create directories if they don't exist
    os.makedirs("runs/mwrl_experiment", exist_ok=True)
    os.makedirs("checkpoints/mwrl_experiment", exist_ok=True)
    os.makedirs("results/mwrl_experiment", exist_ok=True)
    
    cmd = [
        sys.executable, "train_mwrl_distributed_new.py",
        "--batch-size", "128",
        "--epochs", "100", 
        "--lr", "0.1",
        "--nesting-list", "8,16,32,64,128,256,512,1024,2048",
        "--alpha", "1.0",                    # NEW: MRL loss weight (set to 0.0 for Wasserstein-only)
        "--wasserstein-weight", "1.0",
        "--beta", "1.0",
        "--epsilon", "0.1",                  # NEW: Sinkhorn regularization
        "--eval-interval", "5",
        "--multidim-plot-interval", "10",    # NEW: Multi-dimensional plot frequency
        "--num-gpus", "4",
        "--log-dir", "runs/mwrl_alpha_1_lambda_1.0_beta_1.0_epsilon_0.1_bs_128_v1",
        "--checkpoint-dir", "checkpoints/mwrl_alpha_1_lambda_1.0_beta_1.0_epsilon_0.1_bs_128_v1",
        "--csv-dir", "results/mwrl_alpha_1_lambda_1.0_beta_1.0_epsilon_0.1_bs_128_v1",  # NEW: CSV output directory
        "--grad-clip", "1.0",
        "--label-smoothing", "0.1",
        "--pretrained"                       # RECOMMENDED: Use pretrained ResNet50
    ]
    
    print("="*70)
    print("MWRL Training Configuration")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nCommand:")
    print(" ".join(cmd))
    print("\n" + "="*70)
    print("Outputs will be saved to:")
    print(f"  - TensorBoard logs:  runs/mwrl_experiment/")
    print(f"  - Checkpoints:       checkpoints/mwrl_experiment/")
    print(f"  - CSV results:       results/mwrl_experiment/")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("✓ Training completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. View training progress:")
        print("     tensorboard --logdir runs/mwrl_experiment")
        print("\n  2. Analyze CSV results:")
        print("     python analyze_csv_results.py \\")
        print("         --csv-dir results/mwrl_experiment \\")
        print("         --output-dir paper_plots \\")
        print("         --plot-all --export-latex")
        print("\n  3. Find results:")
        print(f"     - Best model:     checkpoints/mwrl_experiment/best_model.pt")
        print(f"     - Metrics CSV:    results/mwrl_experiment/validation_metrics.csv")
        print(f"     - Loss CSV:       results/mwrl_experiment/training_loss.csv")
        print(f"     - Config JSON:    results/mwrl_experiment/training_config.json")
        print("="*70 + "\n")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = run_mwrl_training()
    sys.exit(exit_code)