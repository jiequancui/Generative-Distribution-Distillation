#!/bin/bash
#SBATCH --job-name=wrn402_shufflev1_lr3e-3_wd1e-2_stage1
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=wrn402_shufflev1_lr3e-3_wd1e-2_stage1.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj192,proj194,proj195,proj196,proj197,proj199,proj202,proj203,proj204


source activate py3.8_pt1.8.1
python train_student_stage1.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 2 --epochs 240 --learning_rate 3e-3 --weight_decay 1e-2 --batch_size 64 
