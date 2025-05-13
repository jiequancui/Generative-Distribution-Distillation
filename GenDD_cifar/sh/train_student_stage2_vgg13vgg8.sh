#!/bin/bash
#SBATCH --job-name=vgg13vgg8_stage2_w09_lr2e-3_steplr_mix_t3
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=vgg13vgg8_stage2_w09_lr2e-3_steplr_mix_t3.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj195


source activate py3.8_pt1.8.1
python train_student_stage22.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s vgg8 --trial 3 --epochs 240 --learning_rate 2e-3 --weight_decay 5e-2 --batch_size 64  --resume save/student_model/S:vgg8_T:vgg13_cifar100_stage1_1/vgg8_best.pth --smooth 0.9

#4 0.7 5 0.6 6 0.5



