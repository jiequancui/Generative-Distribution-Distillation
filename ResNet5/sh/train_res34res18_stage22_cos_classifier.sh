#!/bin/bash
#SBATCH --job-name=res34res18_lr2e-3_wd2e-2_stage22_1024_mix_w07_bt512_cos_mul5_classifier
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr2e-3_wd2e-2_stage22_1024_mix_w07_bt512_cos_mul5_classifier
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj195

source activate py3.8_pt1.8.1
python main_distill_stage22_cosine.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 1e-2 \
	       --weight-decay 1e-4 \
	       --finetune_classifier \
	       --teacher_arch resnet34 \
	       --smooth 0.7 \
	       --beta 1.0 \
	       --diffusion_batch_mul 50 \
	       --diffloss_w 1024 \
	       --warmup_epochs 0 \
	       --epochs 10 \
	       -j 8 \
	       -b 512 \
	       --mark 'workdir/res34res18_lr2e-3_wd2e-2_stage22_1024_mix_w07_bt512_cos_mul5_classifier' \
	       --reload workdir/res34res18_lr2e-3_wd2e-2_stage22_1024_mix_w07_bt512_cos_mul5/model_best.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet

