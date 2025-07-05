#!/bin/bash
#SBATCH --job-name=cifar100_res32x4_res8x4_gendd
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=cifar100_res32x4_res8x4_gendd.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj193

source activate py3.8_pt1.8.1
python train_student_pretrain.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 5e-2 --batch_size 64 --epochs 50

python train_student_gendd.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 99 --epochs 240 --learning_rate 1e-3 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:resnet8x4_T:resnet32x4_cifar100_stage1_1/resnet8x4_last.pth --smooth 0.7 --warmup_epochs 50  --diffusion_batch_mul 30
