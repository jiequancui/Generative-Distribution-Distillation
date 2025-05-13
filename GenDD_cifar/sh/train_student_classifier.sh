#!/bin/bash
#SBATCH --job-name=res8x4_res8x4
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res8x4_res8x4.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194


source activate py3.8_pt1.8.1
python train_student.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 1 --epochs 100 --learning_rate 1e-3 --weight_decay 0.0 --batch_size 256 --finetune_classifier --resume save/student_model/S:resnet8x4_T:resnet32x4_cifar100_3/resnet8x4_best.pth 
