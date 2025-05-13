#!/bin/bash
#SBATCH --job-name=resnet8x4_adam_240e_lr3e-3_wd5e-2_wd
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=resnet8x4_adam_240e_lr3e-3_wd5e-2_wd.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194


source activate py3.8_pt1.8.1

python train_teacher_wadam.py --model resnet8x4 --learning_rate 3e-3 --weight_decay 5e-2
