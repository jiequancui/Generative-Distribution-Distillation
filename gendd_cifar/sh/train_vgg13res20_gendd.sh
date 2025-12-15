#!/bin/bash
#SBATCH --job-name=cifar100_vgg13res20_gendd
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=cifar100_vgg13res20_gendd.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj199


source activate py3.8_pt1.8.1
# optional pretraining
python train_student_pretrain.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s resnet20 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 5e-2 --batch_size 64 --epochs 50

python train_student_gendd.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s resnet20 --trial 99 --epochs 240 --learning_rate 1e-3 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:resnet20_T:vgg13_cifar100_stage1_1/resnet20_best.pth --smooth 0.9 --warmup_epochs 50 --diffusion_batch_mul 30
