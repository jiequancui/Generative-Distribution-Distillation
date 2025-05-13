#!/bin/bash
#SBATCH --job-name=wrn402_wrn401_stage1
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=wrn402_wrn401_stage1.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj193,proj194


source activate py3.8_pt1.8.1
python train_student_stage1.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 5e-2 --batch_size 64 
