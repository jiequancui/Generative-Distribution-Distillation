#!/bin/bash
#SBATCH --job-name=wrn402_wrn162_stage2_w08_lr2e-3_steplr_mix_t4
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=wrn402_wrn162_stage2_w08_lr2e-3_steplr_mix_t4.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj194


source activate py3.8_pt1.8.1
python train_student_stage22.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --trial 4 --epochs 240 --learning_rate 2e-3 --weight_decay 5e-2 --batch_size 128 --resume save/student_model/S:wrn_16_2_T:wrn_40_2_cifar100_stage1_1/wrn_16_2_best.pth --smooth 0.8
