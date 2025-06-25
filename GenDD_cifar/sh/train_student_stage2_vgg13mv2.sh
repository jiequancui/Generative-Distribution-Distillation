#!/bin/bash
#SBATCH --job-name=vgg13mv2_w08_lr1e-4_wd1e-2_bt128_pretrain50e_warm50_beta095_mul30_cos_t105
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=vgg13mv2_w08_lr1e-4_wd1e-2_bt128_pretrain50e_warm50_beta095_mul30_cos_t105.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj203

source activate py3.8_pt1.8.1
# optional pretraining
python train_student_stage1.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --trial 1 --epochs 240 --learning_rate 3e-3 --weight_decay 1e-2 --batch_size 64 --epochs 50 

python train_student_gendd.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --trial 105 --epochs 240 --learning_rate 1e-4 --weight_decay 1e-2 --batch_size 128 --resume save/student_model/S:MobileNetV2_T:vgg13_cifar100_stage1_1/MobileNetV2_last.pth --smooth 0.8 --diffusion_batch_mul 30 --warmup_epochs 50 --cos
