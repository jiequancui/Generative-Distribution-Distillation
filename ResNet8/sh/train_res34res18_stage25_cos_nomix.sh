#!/bin/bash
#SBATCH --job-name=res34res18_lr2e-3_wd2e-2_stage25_1024_nomix_w00_bt512_cos_mul5
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr2e-3_wd2e-2_stage25_1024_nomix_w00_bt512_cos_mul5.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj202

source activate py3.8_pt1.8.1
python main_distill_stage25_cosine_nomix.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
	       --teacher_arch resnet34 \
	       --smooth 0.0 \
	       --beta 1.0 \
	       --diffusion_batch_mul 5 \
	       --diffloss_w 1024 \
	       --warmup_epochs 10 \
	       --epochs 100 \
	       -j 64 \
	       -b 512 \
	       --mark 'workdir/res34res18_lr2e-3_wd2e-2_stage25_1024_nomix_w00_bt512_cos_mul5' \
               --reload workdir/res34res18_lr5e-3_wd2e-2_stage1/model_best.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
