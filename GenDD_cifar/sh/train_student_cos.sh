#!/bin/bash
#SBATCH --job-name=res32x4_res8x4_w08_lr5e-3_wd4e-2_bt64_480e_cos
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res32x4_res8x4_w08_lr5e-3_wd4e-2_bt64_480e_cos.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194

source activate py3.8_pt1.8.1
python train_student_cos.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 10 --epochs 480 --learning_rate 5e-3 --weight_decay 4e-2 --batch_size 64 
