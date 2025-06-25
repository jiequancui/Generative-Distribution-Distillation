#!/bin/bash
#SBATCH --job-name=res50mv1_lr2e-3_wd2e-2_stage26_1024_mix_w00_token64_drop02_bt512_cos_mul10_adam_beta_min096_new
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50mv1_lr2e-3_wd2e-2_stage26_1024_mix_w00_token64_drop02_bt512_cos_mul10_adam_beta_min096_new.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj201

source activate py3.8_pt1.8.1
python main_distill_stage26_cosine.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8886' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --adam_beta_min 0.96 \
	       --weight-decay 2e-2 \
	       --teacher_arch resnet50 \
	       --smooth 0.0 \
	       --beta 1.0 \
	       --token_dim 64 \
	       --cond_drop_prob 0.2 \
	       --diffusion_batch_mul 10 \
	       --diffloss_w 1024 \
	       --diff_weight_decay 0 \
	       --warmup_epochs 20 \
	       --epochs 100 \
	       -j 64 \
	       -b 512 \
               --mark 'workdir/res50mv1_lr2e-3_wd2e-2_stage26_1024_mix_w00_token64_drop02_bt512_cos_mul10_adam_beta_min096_new' \
	       /mnt/proj198/jqcui/Data/ImageNet
