#!/bin/bash
#SBATCH --job-name=res56_res20_stage1
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res56_res20_stage1.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194


source activate py3.8_pt1.8.1
python train_student_stage1.py --path_t save/models/resnet56_vanilla/ckpt_epoch_240.pth --model_s resnet20 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 5e-2 --batch_size 64 
