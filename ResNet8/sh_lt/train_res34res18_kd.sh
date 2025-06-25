#!/bin/bash
#SBATCH --job-name=imagenetlt_kd_res34res18_lr2e-3_wd2e-2_bt256_cos_w10
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenetlt_kd_res34res18_lr2e-3_wd2e-2_bt256_cos_w10.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj197

source activate py3.8_pt1.8.1


python main_kd_lt.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8889' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
	       --teacher_arch resnet34 \
	       --teacher_model \
	       --distill_w 1.0 \
	       --cos \
	       --epochs 100 \
	       --warmup_epochs 10 \
	       -j 64 \
	       -b 256 \
               --mark 'workdir_lt/imagenetlt_kd_res34res18_lr2e-3_wd2e-2_bt256_w10' \
	       /mnt/proj198/jqcui/Data/ImageNet
