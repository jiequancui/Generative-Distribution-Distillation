#!/bin/bash
#SBATCH --job-name=wrn402_wrn162_w08_lr1e-3_wd5e-2_bt64_pretrain50e_warm50_beta095_mul30_t99
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=wrn402_wrn162_w08_lr1e-3_wd5e-2_bt64_pretrain50e_warm50_beta095_mul30_t99.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj199


source activate py3.8_pt1.8.1

# optional pretraining
python train_student_stage1.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 5e-2 --batch_size 64 --epochs 50

python train_student_gendd.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --trial 99 --epochs 240 --learning_rate 1e-3 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:wrn_16_2_T:wrn_40_2_cifar100_stage1_1/wrn_16_2_last.pth --smooth 0.8 --diffusion_batch_mul 30 --warmup_epochs 50
