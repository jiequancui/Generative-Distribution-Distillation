#!/bin/bash
#SBATCH --job-name=wrn402_shufflev1_stage2_w07_lr1e-5_wd1e-2_bt128_pretrain50e_warm50_beta095_mul30_step_nodecay_t107
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=wrn402_shufflev1_stage2_w07_lr1e-5_wd1e-2_bt128_pretrain50e_warm50_beta095_mul30_step_nodecay_t107.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj199


source activate py3.8_pt1.8.1
python train_student_stage1.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 1e-2 --batch_size 128 --epochs 50

python train_student_gendd.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 107 --epochs 240 --learning_rate 1e-5 --weight_decay 1e-2 --batch_size 128 --resume save/student_model/S:ShuffleV1_T:wrn_40_2_cifar100_stage1_1/ShuffleV1_last.pth --smooth 0.7 --diff_weight_decay 0 --diffusion_batch_mul 30 --warmup_epochs 50
