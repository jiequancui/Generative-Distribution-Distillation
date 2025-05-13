#!/bin/bash
#SBATCH --job-name=res34res18_lr5e-4_wd5e-2_stage23_1024_nomix_w08
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr5e-4_wd5e-2_stage23_1024_nomix_w08.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -x proj202,proj196,proj199,proj195,proj204,proj194,proj192

source activate py3.8_pt1.8.1
python main_distill_stage23.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8888' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 5e-4 \
	       --weight-decay 5e-2 \
	       --teacher_arch resnet34 \
	       --smooth 0.8 \
	       --mark 'workdir/res34res18_lr5e-4_wd5e-2_stage23_1024_nomix_w08' \
	       --reload workdir/res34res18_lr1e-3_wd5e-2_stage1/model_best.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
