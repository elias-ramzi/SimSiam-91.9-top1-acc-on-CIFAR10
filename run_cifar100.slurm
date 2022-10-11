#!/bin/bash
#SBATCH --job-name=C100     # job name
#SBATCH --nodes=1
#SBATCH --ntasks=4                   # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%A_%a.out # output file name
#SBATCH --error=logs/%A_%a.err  # error file name
#SBATCH --array=0

set -x
cd $WORK/SimSiam-91.9-top1-acc-on-CIFAR10

module purge
module load pytorch-gpu/py3/1.11.0

export TORCH_HOME=${SCRATCH}/pretrained_models
export TMPDIR=$JOBSCRATCH
export TQDM_DISABLE='yes'


arch[0]=resnet18
arch[1]=resnet34
arch[2]=resnet50


srun python main.py \
--data_root $SCRATCH \
--exp_dir $SCRATCH/ood_experiments/cifar100/${arch[${SLURM_ARRAY_TASK_ID}]} \
--distributed \
--is_cluster \
--dataset cifar100 \
--arch ${arch[${SLURM_ARRAY_TASK_ID}]}
