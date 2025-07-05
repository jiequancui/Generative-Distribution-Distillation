#!/bin/bash
#SBATCH --job-name=cifar100_wrn402_wrn401_gendd
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=cifar100_wrn402_wrn401_gendd.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj199


source activate py3.8_pt1.8.1

# optional pretraining
python train_student_pretrain.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 5e-2 --batch_size 64 --epochs 50

python train_student_gendd.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --trial 99 --epochs 240 --learning_rate 1e-3 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:wrn_40_1_T:wrn_40_2_cifar100_stage1_1/wrn_40_1_last.pth --smooth 0.8 --warmup_epochs 50 --diffusion_batch_mul 30
