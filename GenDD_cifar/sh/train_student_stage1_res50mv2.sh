#!/bin/bash
#SBATCH --job-name=res50_mv2_lr3e-3_wd1e-2_stage1
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50_mv2_lr3e-3_wd1e-2_stage1.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj198

source activate py3.8_pt1.8.1
python train_student_stage1.py --path_t save/models/ResNet50_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --trial 2 --epochs 240 --learning_rate 3e-3 --weight_decay 1e-2 --batch_size 64 
