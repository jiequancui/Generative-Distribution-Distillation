#!/bin/bash
#SBATCH --job-name=arinar_lr_3e-3
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=lr_3e-3.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194


source activate py3.8_pt1.8.1


python train_student.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1 --learning_rate 2e-3

