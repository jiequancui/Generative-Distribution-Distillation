#!/bin/bash
#SBATCH --job-name=res110_res32_w10_lr3e-3_wd5e-2_bt64_240e_decay
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res110_res32_w10_lr3e-3_wd5e-2_bt64_240e_decay.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194


source activate py3.8_pt1.8.1
python train_student_wd.py --path_t save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet32 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 5e-2 --batch_size 64 
