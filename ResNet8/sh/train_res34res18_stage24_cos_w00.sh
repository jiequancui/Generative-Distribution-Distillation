#!/bin/bash
#SBATCH --job-name=res34res18_lr2e-3_wd2e-2_stage24_1024_mix_w00_token64_drop02_crop008_alpha10_bt512_cos_mul10_adam_beta_min096
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr2e-3_wd2e-2_stage24_1024_mix_w00_token64_drop02_crop008_alpha10_bt512_cos_mul10_adam_beta_min096.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj201

source activate py3.8_pt1.8.1


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
	       --smooth 0.0 \
	       --alpha 1.0 \
	       --beta 1.0 \
	       --token_dim 64 \
	       --cond_drop_prob 0.2 \
	       --crop_scale 0.08 \
	       --cos \
	       --diffusion_batch_mul 10 \
	       --diffloss_w 1024 \
	       --diff_weight_decay 0 \
	       --epochs 100 \
	       --warmup_epochs 10 \
	       -j 64 \
	       -b 512 \
               --mark 'workdir/res34res18_lr2e-3_wd2e-2_stage24_1024_mix_w00_token64_drop02_crop008_alpha10_bt512_cos_mul10_adam_beta_min096' \
	       /mnt/proj198/jqcui/Data/ImageNet
