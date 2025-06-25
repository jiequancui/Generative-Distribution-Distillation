#!/bin/bash
#SBATCH --job-name=res50_mv2_w08_lr1e-4_wd2e-2_bt128_pretrain50e_warm50_beta095_mul10_cos_t100
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50_mv2_w08_lr1e-4_wd2e-2_bt128_pretrain50e_warm50_beta095_mul10_cos_t100.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj199

source activate py3.8_pt1.8.1

python train_student_stage1.py --path_t save/models/ResNet50_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 2e-2 --batch_size 64 --epochs 50

python train_student_gendd.py --path_t save/models/ResNet50_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --trial 100 --epochs 240 --learning_rate 1e-4 --weight_decay 2e-2 --batch_size 128 --resume save/student_model/S:MobileNetV2_T:ResNet50_cifar100_stage1_1/MobileNetV2_last.pth --smooth 0.8 --diff_weight_decay 2e-2 --diffusion_batch_mul 10 --warmup_epochs 50 --cos
