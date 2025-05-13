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


#python train_student_stage22.py --path_t save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet32 --trial 3 --epochs 800 --learning_rate 1e-3 --weight_decay 1e-3 --batch_size 256 --evaluate --resume save/student_model/S\:resnet32_T\:resnet110_cifar100_stage2_3/resnet32_best.pth --evaluate 

#python train_student_stage22.py --path_t save/models/resnet56_vanilla/ckpt_epoch_240.pth --model_s resnet20 --trial 3 --epochs 240 --learning_rate 5e-4 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:resnet20_T:resnet56_cifar100_stage2_3/resnet20_best.pth --evaluate

#python train_student_stage22.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --trial 5 --epochs 240 --learning_rate 5e-4 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:wrn_16_2_T:wrn_40_2_cifar100_stage2_5/wrn_16_2_best.pth --evaluate

#python train_student_stage22.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --trial 2 --epochs 240 --learning_rate 5e-4 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:wrn_40_1_T:wrn_40_2_cifar100_stage2_5/wrn_40_1_best.pth --evaluate

#python train_student_stage22.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s vgg8 --trial 8 --epochs 240 --learning_rate 5e-4 --weight_decay 5e-2 --batch_size 64  --resume save/student_model/S:vgg8_T:vgg13_cifar100_stage2_8/vgg8_best.pth --evaluate

#python train_student_stage22.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 4 --epochs 240 --learning_rate 5e-4 --weight_decay 5e-2 --batch_size 64 --resume save/student_model/S:resnet8x4_T:resnet32x4_cifar100_stage2_4/resnet8x4_best.pth --evaluate

#python train_student_stage22.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 5 --epochs 240 --learning_rate 1e-5 --weight_decay 1e-2 --batch_size 64 --resume save/student_model/S:ShuffleV1_T:resnet32x4_cifar100_stage2_5/ShuffleV1_best.pth --evaluate 


#python train_student_stage22.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV2 --trial 2 --epochs 240 --learning_rate 2e-5 --weight_decay 1e-2 --batch_size 64 --resume save/student_model/S:ShuffleV2_T:resnet32x4_cifar100_stage2_2/ShuffleV2_best.pth --evaluate

#python train_student_stage22.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 3 --epochs 240 --learning_rate 2e-5 --weight_decay 1e-2 --batch_size 128 --resume save/student_model/S:ShuffleV1_T:wrn_40_2_cifar100_stage2_3/ShuffleV1_best.pth --evaluate

python train_student_stage22.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --trial 5 --epochs 240 --learning_rate 1e-5 --weight_decay 1e-2 --batch_size 128 --resume save/student_model/S:MobileNetV2_T:vgg13_cifar100_stage2_2/MobileNetV2_best.pth --smooth 0.75 --diffloss_w 1280 --evaluate 
