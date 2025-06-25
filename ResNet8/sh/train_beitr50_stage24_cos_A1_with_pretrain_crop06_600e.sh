#!/bin/bash
#SBATCH --job-name=beitr50_stage24_lr2e-3_wd2e-2_w09_mix0_1024_600e_crop06_rand_mixup02_cutmix10_smoothing01_pretrain20e
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=beitr50_stage24_lr2e-3_wd2e-2_w09_mix0_1024_600e_crop06_rand_mixup02_cutmix10_smoothing01_pretrain20e.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj204

source activate py3.8_pt1.8.1
#python main_distill_stage1.py -a resnet50 \
#	       --dist-url 'tcp://127.0.0.1:8888' \
#               --dist-backend 'nccl' \
#	       --multiprocessing-distributed \
#	       --world-size 1 \
#	       --rank 0 \
#	       --lr 2e-3 \
#	       --weight-decay 2e-2 \
#               --teacher_arch beitv2_base_patch16_224 \
#	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_base_patch16_224_pt1k_ft21kto1k_new.pth \
#	       --mixup 0.2 \
#	       --cutmix 1.0 \
#	       --smoothing 0.1 \
#               --aug_type 'rand' \
#               -j 64 \
#	       -b 512 \
#	       --epochs 20 \
#	       --mark 'workdir/beitres50_stage1_20e_A0' \
#	       /mnt/proj198/jqcui/Data/ImageNet


python main_distill_stage24_cosine.py -a resnet50 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
               --teacher_arch beitv2_base_patch16_224 \
	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_base_patch16_224_pt1k_ft21kto1k_new.pth \
	       --smooth 0.9 \
	       --beta 0.0 \
	       --crop_scale 0.6 \
	       --diffusion_batch_mul 10 \
	       --diffloss_w 1024 \
	       --warmup_epochs 20 \
	       --epochs 600 \
	       --mixup 0.2 \
	       --cutmix 1.0 \
	       --smoothing 0.1 \
               --aug_type 'rand' \
               --cos \
	       -j 64 \
	       -b 512 \
               --reload workdir/beitres50_stage1_20e_A0/checkpoint.pth.tar \
	       --mark 'workdir/beitr50_stage24_lr2e-3_wd2e-2_w09_mix0_1024_600e_crop06_rand_mixup02_cutmix10_smoothing01_A0_pretrain20e' \
	       /mnt/proj198/jqcui/Data/ImageNet
