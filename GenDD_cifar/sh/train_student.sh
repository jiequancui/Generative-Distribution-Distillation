#!/bin/bash
#SBATCH --job-name=res32x4_res8x4_w00_lr3e-3_wd5e-2_bt64_240e_weight5
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res32x4_res8x4_w00_lr3e-3_wd5e-2_bt64_240e_weight5.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194

source activate py3.8_pt1.8.1
python train_student_wd.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 100 --epochs 240 --learning_rate 3e-3 --weight_decay 5e-2 --batch_size 64 
