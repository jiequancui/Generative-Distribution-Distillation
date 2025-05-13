#!/bin/bash
#SBATCH --job-name=wrn402_shufflev1_stage2_w07_lr1e-5_wd1e-2_steplr_mix_t4
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=wrn402_shufflev1_stage2_w07_lr1e-5_wd1e-2_steplr_mix_t4.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj195,proj194


source activate py3.8_pt1.8.1
python train_student_stage22.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 4 --epochs 240 --learning_rate 1e-5 --weight_decay 1e-2 --batch_size 128 --resume save/student_model/S:ShuffleV1_T:wrn_40_2_cifar100_stage1_2/ShuffleV1_best.pth --smooth 0.7 --warmup_epochs 10 --diff_weight_decay 1e-2
