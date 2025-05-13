#!/bin/bash
#SBATCH --job-name=res32x4_shufflev1_stage2_w07_lr2e-5_wd1e-2_bt128_steplr_mix_t5
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res32x4_shufflenetv1_stage2_w07_lr2e-5_wd1e-2_bt128_steplr_mix_t5.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj195,proj194

source activate py3.8_pt1.8.1
python train_student_stage22.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 5 --epochs 240 --learning_rate 2e-5 --weight_decay 1e-2 --batch_size 256 --resume save/student_model/S:ShuffleV1_T:resnet32x4_cifar100_stage1_2/ShuffleV1_best.pth --smooth 0.7 --warmup_epochs 10 --diff_weight_decay 1e-2
