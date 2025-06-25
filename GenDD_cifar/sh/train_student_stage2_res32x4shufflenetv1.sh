#!/bin/bash
#SBATCH --job-name=res32x4_shufflev1_w07_lr1e-5_wd1e-2_bt128_pretrain20e_warm20_beta095_mul30_step_nodecay_t103
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res32x4_shufflev1_w07_lr1e-5_wd1e-2_bt128_pretrain20e_warm20_beta095_mul30_step_nodecay_t103.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -w proj199

source activate py3.8_pt1.8.1
# optional pretraining
python train_student_stage1.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 1 --epochs 240 --learning_rate 1e-5 --weight_decay 1e-2 --batch_size 128 --epochs 20

python train_student_gendd.py --path_t save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --trial 103 --epochs 240 --learning_rate 1e-5 --weight_decay 1e-2 --batch_size 128 --resume save/student_model/S:ShuffleV1_T:resnet32x4_cifar100_stage1_1/ShuffleV1_last.pth --smooth 0.7 --diff_weight_decay 0 --diffusion_batch_mul 30 --warmup_epochs 20
