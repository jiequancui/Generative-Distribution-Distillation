#!/bin/bash
#SBATCH --job-name=vgg13mv2_stage2_w08_lr1e-3_wd1e-2_bt128_mul100_steplr_mix_t5
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=vgg13mv2_stage2_w08_lr1e-3_wd1e-2_bt128_mul100_steplr_mix_t5.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj195,proj194

source activate py3.8_pt1.8.1
python train_student_stage22.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --trial 5 --epochs 240 --learning_rate 1e-3 --weight_decay 1e-2 --batch_size 128 --resume save/student_model/S:MobileNetV2_T:vgg13_cifar100_stage1_2/MobileNetV2_best.pth --smooth 0.8 --diffloss_w 1280 --diff_weight_decay 1e-2 --adam_beta 0.95 --diffusion_batch_mul 100 --cos
