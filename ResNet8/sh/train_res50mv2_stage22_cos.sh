#!/bin/bash
#SBATCH --job-name=res50mv1_lr2e-3_wd5e-3_stage22_1024_mix_w07_token64_drop02_bt512_cos_mul5_adam_beta_min095
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50mv1_lr2e-3_wd5e-3_stage22_1024_mix_w07_token64_drop02_bt512_cos_mul5_adam_beta_min095.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj200

source activate py3.8_pt1.8.1
python main_distill_stage22_cosine.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8886' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --adam_beta_min 0.95 \
	       --weight-decay 5e-3 \
	       --teacher_arch resnet50 \
	       --smooth 0.7 \
	       --beta 1.0 \
	       --token_dim 64 \
	       --cond_drop_prob 0.2 \
	       --diffusion_batch_mul 5 \
	       --diffloss_w 1024 \
	       --diff_weight_decay 0 \
	       --warmup_epochs 10 \
	       --epochs 100 \
	       -j 64 \
	       -b 512 \
               --mark 'workdir/res50mv1_lr2e-3_wd5e-3_stage22_1024_mix_w07_token64_drop02_bt512_cos_mul5_adam_beta_min095' \
               --reload workdir/res50mv1_lr1e-3_wd5e-2_stage1/model_best.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
