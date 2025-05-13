#!/bin/bash
#SBATCH --job-name=res32x4_shufflenetv1_lr_3e-3_wd1e-2_stage1
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res32x4_shufflenetv1_lr3e-3_wd1e-2_stage1.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj193

source activate py3.8_pt1.8.1
python train_student_stage1.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 2 --epochs 240 --learning_rate 3e-3 --weight_decay 1e-2 --batch_size 64 
