#!/bin/bash
#SBATCH --job-name=MWRL_training
#SBATCH --partition=gpu_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/home05/kumartanmay/main/MatryoWass/logs/mwrl_%j.out
#SBATCH --error=/n/home05/kumartanmay/main/MatryoWass/logs/mwrl_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kumartanmay@g.harvard.edu

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""
echo "=========================================="
echo ""

# Load necessary modules (adjust based on your cluster if needed)
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment
echo "Activating virtual environment..."
source /n/home05/kumartanmay/env2/bin/activate

# Verify activation
echo "Virtual environment activated: $VIRTUAL_ENV"
echo ""

# Change to your project directory
cd /n/home05/kumartanmay/main/MatryoWass

# Print Python and PyTorch information
echo "Python version:"
python --version
echo ""
echo "Python path:"
which python
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
echo ""
echo "=========================================="
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the training script
echo "Starting MWRL training..."
echo "=========================================="
echo ""

python /n/home05/kumartanmay/main/MatryoWass/run_mwrl_training_new.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Deactivate virtual environment
deactivate

# Print results summary
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  - Logs:        /n/home05/kumartanmay/main/MatryoWass/logs/mwrl_${SLURM_JOB_ID}.out"
    echo "  - Checkpoints: /n/home05/kumartanmay/main/MatryoWass/checkpoints/mwrl_experiment/"
    echo "  - CSV files:   /n/home05/kumartanmay/main/MatryoWass/results/mwrl_experiment/"
    echo "  - TensorBoard: /n/home05/kumartanmay/main/MatryoWass/runs/mwrl_experiment/"
    echo ""
    echo "To analyze results:"
    echo "  cd /n/home05/kumartanmay/main/MatryoWass"
    echo "  source /n/home05/kumartanmay/env2/bin/activate"
    echo "  python analyze_csv_results.py --csv-dir results/mwrl_experiment --plot-all"
else
    echo ""
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo "Check logs at:"
    echo "  - Output: /n/home05/kumartanmay/main/MatryoWass/logs/mwrl_${SLURM_JOB_ID}.out"
    echo "  - Errors: /n/home05/kumartanmay/main/MatryoWass/logs/mwrl_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE