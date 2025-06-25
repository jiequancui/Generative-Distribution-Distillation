#!/bin/bash
#SBATCH --job-name=res34res18_lr2e-3_wd2e-2_stage24_1024_mix_w06_token64_drop02_crop06_alpha10_bt512_cos_mul10_adam_beta_min096_pretrain20e_warm10
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr2e-3_wd2e-2_stage24_1024_mix_w06_token64_drop02_crop06_alpha10_bt512_cos_mul10_adam_beta_min096_pretrain20e_warm10.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj200

source activate py3.8_pt1.8.1

#python main_distill_stage1.py -a resnet18 \
#	       --dist-url 'tcp://127.0.0.1:8888' \
#               --dist-backend 'nccl' \
#	       --multiprocessing-distributed \
#	       --world-size 1 \
#	       --rank 0 \
#	       --lr 2e-3 \
#	       --weight-decay 2e-2 \
#	       --teacher_arch resnet34 \
#	       -j 64 \
#	       -b 512 \
#	       --epochs 20 \
#	       --mark 'workdir/res34res18_stage1_20e' \
#	       /mnt/proj198/jqcui/Data/ImageNet


python main_distill_stage24_cosine.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8889' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
	       --adam_beta_min 0.96 \
	       --teacher_arch resnet34 \
	       --smooth 0.6 \
	       --alpha 1.0 \
	       --beta 1.0 \
	       --token_dim 64 \
	       --cond_drop_prob 0.2 \
	       --crop_scale 0.6 \
	       --cos \
	       --diffusion_batch_mul 10 \
	       --diffloss_w 1024 \
	       --diff_weight_decay 0 \
	       --epochs 100 \
	       --warmup_epochs 10 \
	       -j 64 \
	       -b 512 \
               --mark 'workdir/res34res18_lr2e-3_wd2e-2_stage24_1024_mix_w06_token64_drop02_crop06_alpha10_bt512_cos_mul10_adam_beta_min096_pretrain20e_warm10' \
	       --reload workdir/res34res18_stage1_20e/checkpoint.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
