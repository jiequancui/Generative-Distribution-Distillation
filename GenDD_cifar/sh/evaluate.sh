#!/bin/bash
#SBATCH --job-name=cifar100_evaluation
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=cifar100_evaluation.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77,proj194


source activate py3.8_pt1.8.1

# w09
#python train_student_gendd.py --path_t save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet32 --trial 2 --batch_size 256 --resume save/student_model/S:resnet32_T:resnet110_cifar100_stage2_102/resnet32_best.pth --evaluate --num_sampling 5

#w09
#python train_student_gendd.py --path_t save/models/resnet56_vanilla/ckpt_epoch_240.pth --model_s resnet20 --trial 2 --batch_size 64 --resume save/student_model/S:resnet20_T:resnet56_cifar100_stage2_99/resnet20_best.pth --evaluate --num_sampling 5

# w08
#python train_student_gendd.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --trial 2 --batch_size 64 --resume save/student_model/S:wrn_16_2_T:wrn_40_2_cifar100_stage2_99/wrn_16_2_best.pth --evaluate --num_sampling 5

# w08
#python train_student_gendd.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --trial 2 --batch_size 64 --resume save/student_model/S:wrn_40_1_T:wrn_40_2_cifar100_stage2_99/wrn_40_1_best.pth --evaluate --num_sampling 5

# w07
#python train_student_gendd.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s vgg8 --trial 2 --batch_size 64 --resume save/student_model/S:vgg8_T:vgg13_cifar100_stage2_99/vgg8_best.pth --evaluate --num_sampling 5

# w07
#python train_student_gendd.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 2 --batch_size 64 --resume save/student_model/S:resnet8x4_T:resnet32x4_cifar100_stage2_99/resnet8x4_best.pth --evaluate --num_sampling 5


###############
#python train_student_gendd.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 2 --batch_size 64 --resume save/student_model/S:ShuffleV1_T:resnet32x4_cifar100_stage2_116/ShuffleV1_best.pth --evaluate --num_sampling 5


#python train_student_gendd.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV2 --trial 2 --batch_size 64 --resume save/student_model/S:ShuffleV2_T:resnet32x4_cifar100_stage2_113/ShuffleV2_best.pth --evaluate --num_sampling 5

#python train_student_gendd.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 2 --batch_size 64 --resume save/student_model/S:ShuffleV1_T:wrn_40_2_cifar100_stage2_124/ShuffleV1_best.pth --evaluate --num_sampling 5

#python train_student_gendd.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --trial 2 --batch_size 64 --resume save/student_model/S:MobileNetV2_T:vgg13_cifar100_stage2_105/MobileNetV2_best.pth --evaluate --num_sampling 5

#python train_student_gendd.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s vgg8 --trial 2 --batch_size 64 --resume save/student_model/S:vgg8_T:resnet32x4_cifar100_stage2_118/vgg8_best.pth --evaluate --num_sampling 5

#python train_student_gendd.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --trial 2 --batch_size 64 --resume save/student_model/S:wrn_16_2_T:resnet32x4_cifar100_stage2_120/wrn_16_2_best.pth --evaluate --num_sampling 5

#w07
#python train_student_gendd.py --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --trial 2 --batch_size 64 --resume save/student_model/S:resnet8x4_T:wrn_40_2_cifar100_stage2_99/resnet8x4_best.pth --evaluate --num_sampling 5

#w09
#python train_student_gendd.py --path_t save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s resnet20 --trial 2 --batch_size 64 --resume save/student_model/S:resnet20_T:vgg13_cifar100_stage2_99/resnet20_best.pth --evaluate --num_sampling 5
