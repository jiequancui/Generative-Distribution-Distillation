#!/bin/bash
#SBATCH --job-name=res34res18_lr2e-3_wd2e-2_stage22_1024_mix_w07_bt512_cos_mul5_300e
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr2e-3_wd2e-2_stage22_1024_mix_w07_bt512_cos_mul5_300e.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj204

source activate py3.8_pt1.8.1
python main_distill_stage22_cosine.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8889' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
	       --adam_beta 0.999 \
	       --teacher_arch resnet34 \
	       --smooth 0.7 \
	       --beta 1.0 \
	       --diffusion_batch_mul 5 \
	       --diffloss_w 1024 \
	       --diff_weight_decay 0 \
	       --epochs 100 \
	       --warmup_epochs 10 \
	       -j 64 \
	       -b 512 \
	       --mark 'workdir/tmp' \
	       --reload workdir/res34res18_lr5e-3_wd2e-2_stage1/model_best.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
