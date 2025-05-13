#!/bin/bash
#SBATCH --job-name=res32x4_res8x4_stage2_w08_lr2e-3_steplr_mix_t4
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res32x4_res8x4_stage2_w08_lr2e-3_steplr_mix_t4.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj195

source activate py3.8_pt1.8.1
python train_student_stage22.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 4 --epochs 240 --learning_rate 2e-3 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:resnet8x4_T:resnet32x4_cifar100_stage1_1/resnet8x4_best.pth --smooth 0.8
