#!/bin/bash
#SBATCH --job-name=wrn402_wrn162_w10_lr5e-3_wd4e-2_bt64_240e
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=wrn402_wrn162_w10_lr5e-3_wd4e-2_bt64_240e.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194


source activate py3.8_pt1.8.1
python train_student.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --trial 10 --epochs 240 --learning_rate 5e-3 --weight_decay 4e-2 --batch_size 64 
