#!/bin/bash
#SBATCH --job-name=beitlarger50_stage24_lr2e-3_wd2e-2_w09_mix0_1024_300e_crop06_rand_mixup01_cutmix10_pretrain20e_resume
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=beitlarger50_stage24_lr2e-3_wd2e-2_w09_mix0_1024_300e_crop06_rand_mixup01_cutmix10_pretrain20e_resume.log
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
#               --teacher_arch beitv2_large_patch16_224 \
#	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_large_patch16_224_pt1k_ft21kto1k_new.pth \
#	       --mixup 0.1 \
#	       --cutmix 1.0 \
#	       --smoothing 0.0 \
#               --aug_type 'rand' \
#               -j 64 \
#	       -b 512 \
#	       --epochs 20 \
#	       --mark 'workdir/res50mv1_stage1_20e_A1' \
#	       /mnt/proj198/jqcui/Data/ImageNet


python main_distill_stage24_cosine.py -a resnet50 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
	       --teacher_arch beitv2_large_patch16_224 \
	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_large_patch16_224_pt1k_ft21kto1k_new.pth \
	       --smooth 0.9 \
	       --beta 0.0 \
	       --crop_scale 0.6 \
	       --diffusion_batch_mul 10 \
	       --diffloss_w 1024 \
	       --warmup_epochs 20 \
	       --epochs 300 \
	       --mixup 0.1 \
	       --cutmix 1.0 \
	       --smoothing 0.0 \
               --aug_type 'rand' \
               --cos \
	       -j 64 \
	       -b 512 \
	       --reload 'workdir/res50mv1_stage1_20e_A1/checkpoint.pth.tar' \
	       --mark 'workdir/beitlarger50_stage24_lr2e-3_wd2e-2_w09_mix0_1024_300e_crop06_rand_mixup01_cutmix10_A1_pretrain20e_resume' \
	       --resume workdir/beitlarger50_stage24_lr2e-3_wd2e-2_w09_mix0_1024_300e_crop06_rand_mixup01_cutmix10_A1_pretrain20e/checkpoint.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
