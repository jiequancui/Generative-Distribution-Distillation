#!/bin/bash
#SBATCH --job-name=res50_mv2_stage2_w09_lr1e-5_wd1e-2_bt128_steplr_mix_2048_t3
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50_mv2_stage2_w09_lr1e-5_wd1e-2_bt128_steplr_mix_2048_t3.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj195,proj194

source activate py3.8_pt1.8.1
python train_student_stage22.py --path_t save/models/ResNet50_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --trial 3 --epochs 240 --learning_rate 1e-5 --weight_decay 1e-2 --batch_size 128 --smooth 0.9 --warmup_epochs 10 --resume save/student_model/S:MobileNetV2_T:ResNet50_cifar100_stage1_2/MobileNetV2_best.pth --diffloss_w 2048  --adam_beta 0.95 --diff_weight_decay 1e-2 --cos
