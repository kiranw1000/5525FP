#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --account=PAS2912
#SBATCH --ntasks-per-node=1 
#SBATCH--gres=gpu:a100:1

# The following is an example of a single-processor sequential job that uses $TMPDIR as its working area.
# This batch script copies the script file and input file from the directory the
# qsub command was called from into $TMPDIR, runs the code in $TMPDIR,
# and copies the output file back to the original directory.
#   https://www.osc.edu/supercomputing/batch-processing-at-osc/job-scripts
#
# Move to the directory where the job was submitted
#
cd $SLURM_SUBMIT_DIR
cp -r . $TMPDIR
cd $TMPDIR
module load python/3.12
module load cuda
source ~/5525FP/ENV

pip install -r requirements.txt

python -m bitsandbytes

nvidia-smi

python training.py --model_name "Llama-3.2-1B" --dataset_name "5525FP/poisoned-minipile" --output_dir "./results" --learning_rate 1e-3 --batch_size 1 --epochs 10 --weight_decay 0.01 --gradient_accumulation_steps 4 --wandb_key $WANDB_KEY --run_name "poisoned-llm-training" --hf_token $HF_TOKEN

#
# Now, copy data (or move) back once the simulation has completed
#
cp -r results $SLURM_SUBMIT_DIR