#!/bin/bash
#SBATCH --job-name=lr1e-3_wd1e-3_bt256_mul50_400e_new_t3
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=lr1e-3_wd1e-3_bt256_mul50_400e_new_t3.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194


source activate py3.8_pt1.8.1


python train_student.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 3 --epochs 800 --learning_rate 1e-3 --weight_decay 1e-3 --batch_size 256 --evaluate --resume save/student_model/S:resnet8x4_T:resnet32x4_cifar100_1/resnet8x4_best.pth --feature_dim 256 --target_dim 256
