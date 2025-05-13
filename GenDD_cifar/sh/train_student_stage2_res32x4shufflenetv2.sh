#!/bin/bash
#SBATCH --job-name=res32x4_shufflev2_stage2_w08_lr2e-5_wd1e-2_steplr_mix_t2
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res32x4_shufflev2_stage2_w08_lr2e-5_wd1e-2_steplr_mix_t2.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj195,proj194

source activate py3.8_pt1.8.1
python train_student_stage22.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV2 --trial 2 --epochs 240 --learning_rate 2e-5 --weight_decay 1e-2 --batch_size 128 --resume save/student_model/S:ShuffleV2_T:resnet32x4_cifar100_stage1_2/ShuffleV2_best.pth --smooth 0.8 --warmup_epochs 10 --diff_weight_decay 1e-2
