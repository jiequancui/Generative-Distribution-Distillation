#!/bin/bash
#SBATCH --job-name=res34res18_lr5e-4_wd5e-2_stage22_1024_mix02_w07_bt512
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr5e-4_wd5e-2_stage22_1024_mix02_w07_bt512.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj193

source activate py3.8_pt1.8.1
python main_distill_stage22.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8889' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 5e-4 \
	       --weight-decay 5e-2 \
	       --teacher_arch resnet34 \
	       --smooth 0.7 \
	       --beta 0.2 \
	       --diffusion_batch_mul 50 \
	       --diffloss_w 1024 \
	       --warmup_epochs 0 \
	       -j 64 \
	       -b 512 \
	       --mark 'workdir/res34res18_lr5e-4_wd5e-2_stage22_1024_mix02_w07_b512' \
	       --reload workdir/res34res18_lr1e-3_wd5e-2_stage1/model_best.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
