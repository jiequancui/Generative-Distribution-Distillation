#!/bin/bash
#SBATCH --job-name=res34res18_lr2e-3_wd2e-2_stage24_1024_mix_w09_token64_drop02_crop06_alpha10_bt512_cos_mul10_adam_beta_min096_pretrain20e_warm10
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr2e-3_wd2e-2_stage24_1024_mix_w09_token64_drop02_crop06_alpha10_bt512_cos_mul10_adam_beta_min096_pretrain20e_warm10.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj196

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


python main_distill_stage24_cosine_kd.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8889' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 0.2 \
	       --weight-decay 1e-4 \
	       --teacher_arch resnet34 \
	       --cos \
	       --epochs 100 \
	       --warmup_epochs 0 \
	       -j 64 \
	       -b 512 \
               --mark 'workdir/res34res18_lr02_wd1e-4_stage24_cos_warm10_kd' \
	       /mnt/proj198/jqcui/Data/ImageNet
