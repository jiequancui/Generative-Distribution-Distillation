#!/bin/bash
#SBATCH --job-name=imagenetlt_res34res18_lr2e-3_wd2e-2_w09_drop02_crop06_bt256_cos
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenetlt_res34res18_lr2e-3_wd2e-2_w09_drop02_crop06_bt256_cos.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj197

source activate py3.8_pt1.8.1


python main_gendd_lt.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8889' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
	       --adam_beta_min 0.95 \
	       --teacher_arch resnet34 \
	       --smooth 0.9 \
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
	       -b 256 \
               --mark 'workdir_lt/imagenetlt_res34res18_lr2e-3_wd2e-2_w09_drop02_crop06_bt256' \
	       /mnt/proj198/jqcui/Data/ImageNet
