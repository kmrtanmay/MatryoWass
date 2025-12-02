#!/usr/bin/env python3
"""
Quick Multi-Experiment Plotting Script

Simple wrapper for common plotting scenarios.
Edit the experiments list below and run!
"""

import subprocess
import sys
import os


# ============================================================================
# CONFIGURE YOUR EXPERIMENTS HERE
# ============================================================================
# EXPERIMENTS = [
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1_epsilon_0.5_bs_64',
#         'label': 'Batch Size = 64'
#         #'label': 'α=1.0, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'Batch Size = 128'
#         #'label': 'α=0.5, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1_epsilon_0.5_bs_256',
#         'label': 'Batch Size = 256'
#         #'label': 'α=0.0 (W-only), λ=1.0, ε=0.1'
#     },
# ]

# OUTPUT_DIR = 'batch_plots'


# EXPERIMENTS = [
#     {
#         'dir': 'results/mwrl_alpha_0_lambda_1.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'Only Wasserstein Loss'
#         #'label': 'α=1.0, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_0.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'Only Matryoshkha Loss'
#         #'label': 'α=0.5, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'MWRL'
#         #'label': 'α=0.0 (W-only), λ=1.0, ε=0.1'
#     },
# ]
# OUTPUT_DIR = 'loss_plots'

EXPERIMENTS = [
    {
        'dir': 'results/mwrl_alpha_0_lambda_1.0_beta_1_epsilon_0.5_bs_128',
        'label': 'Only Wasserstein Loss'
        #'label': 'α=1.0, λ=1.0, ε=0.1'
    },
    {
        'dir': 'results/mwrl_alpha_1_lambda_0.0_beta_1_epsilon_0.5_bs_128',
        'label': 'Only Matryoshkha Loss'
        #'label': 'α=0.5, λ=1.0, ε=0.1'
    },
    {
        'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1_epsilon_0.5_bs_128',
        'label': 'MWRL'
        #'label': 'α=0.0 (W-only), λ=1.0, ε=0.1'
    },
    {
        'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1.0_epsilon_5.0_bs_128_v1_independent',
        'label': 'Independently Trained FF'
        #'label': 'α=0.0 (W-only), λ=1.0, ε=0.1'
    },
]
OUTPUT_DIR = 'loss_plots_v2'


# EXPERIMENTS = [
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_0.1_beta_1_epsilon_0.5_bs_128',
#         'label': 'λ=0.1'
#         #'label': 'α=1.0, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'λ=1.0'
#         #'label': 'α=0.5, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_10.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'λ=10.0'
#         #'label': 'α=0.0 (W-only), λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_100.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'λ=100.0'
#         #'label': 'α=0.0 (W-only), λ=1.0, ε=0.1'
#     },
# ]
# OUTPUT_DIR = 'lambda_plots'


# EXPERIMENTS = [
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1.0_epsilon_0.1_bs_128_v1',
#         'label': 'ε=0.1'
#         #'label': 'α=1.0, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'ε=0.5'
#         #'label': 'α=0.5, λ=1.0, ε=0.1'
#     },
    
# ]
# OUTPUT_DIR = 'epsilon_plots'

# EXPERIMENTS = [
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_0.1_epsilon_0.5_bs_128',
#         'label': 'β=0.1'
#         #'label': 'α=1.0, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'β=1'
#         #'label': 'α=0.5, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_10.0_epsilon_0.5_bs_128',
#         'label': 'β=10'
#         #'label': 'α=0.5, λ=1.0, ε=0.1'
#     },
    
# ]
# OUTPUT_DIR = 'beta_plots'

# EXPERIMENTS = [
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1_epsilon_0.5_bs_128',
#         'label': 'Trained from Pre-trained Resnet-50 backbone'
#         #'label': 'α=1.0, λ=1.0, ε=0.1'
#     },
#     {
#         'dir': 'results/mwrl_alpha_1_lambda_1.0_beta_1.0_epsilon_0.5_bs_128_scratch',
#         'label': 'Trained from Scratch'
#         #'label': 'α=0.5, λ=1.0, ε=0.1'
#     }
# ]

# OUTPUT_DIR = 'backbone_plots'


# Additional options
DIMENSIONS_TO_PLOT = None  # None = all, or specify like: '8,16,64,256,2048'
FIGURE_SIZE = '12,6'
DPI = 300

# ============================================================================


def run_plotting():
    """Run the multi-experiment plotting script."""
    
    # Extract dirs and labels
    exp_dirs = [exp['dir'] for exp in EXPERIMENTS]
    labels = [exp['label'] for exp in EXPERIMENTS]
    
    # Check if experiments exist
    print("Checking experiment directories...")
    missing = []
    for exp_dir in exp_dirs:
        if not os.path.exists(exp_dir):
            missing.append(exp_dir)
            print(f"  ✗ Missing: {exp_dir}")
        else:
            print(f"  ✓ Found: {exp_dir}")
    
    if missing:
        print(f"\n⚠ Warning: {len(missing)} experiment(s) not found!")
        print("Please update the EXPERIMENTS list in this script.\n")
        return 1
    
    print(f"\n✓ All {len(EXPERIMENTS)} experiments found!\n")
    
    # Build command
    cmd = [
        sys.executable, "plot_multi_experiments.py",
        "--experiment-dirs"] + exp_dirs + [
        "--labels"] + labels + [
        "--output-dir", OUTPUT_DIR,
        "--figsize", FIGURE_SIZE,
        "--dpi", str(DPI)
    ]
    
    if DIMENSIONS_TO_PLOT:
        cmd.extend(["--dimensions", DIMENSIONS_TO_PLOT])
    
    print("Running plotting script...")
    print("Command:", " ".join(cmd))
    print("\n" + "="*70 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("✓ Plotting completed successfully!")
        print("="*70)
        print(f"\nPlots saved to: {OUTPUT_DIR}/")
        print("\nView results:")
        print(f"  - Open {OUTPUT_DIR}/ folder")
        print(f"  - Check results_summary.csv for numerical results")
        print(f"  - Use .pdf files for papers (vector graphics)")
        print("="*70 + "\n")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Plotting failed with error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n✗ Plotting interrupted by user")
        return 1


if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUICK MULTI-EXPERIMENT PLOTTING")
    print("="*70)
    print(f"Experiments to plot: {len(EXPERIMENTS)}")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {exp['label']}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
    exit_code = run_plotting()
    sys.exit(exit_code)