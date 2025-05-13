#!/bin/bash
#SBATCH --job-name=res56_res20_stage2_w10_lr5e-4_steplr_mix_t4
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res56_res20_stage2_w10_lr5e-4_steplr_mix_t4.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj193,proj194


source activate py3.8_pt1.8.1
python train_student_stage22.py --path_t save/models/resnet56_vanilla/ckpt_epoch_240.pth --model_s resnet20 --trial 4 --epochs 240 --learning_rate 5e-4 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:resnet20_T:resnet56_cifar100_stage1_1/resnet20_best.pth --smooth 1.0
