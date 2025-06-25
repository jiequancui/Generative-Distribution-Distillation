#!/bin/bash
#SBATCH --job-name=imagenetlt_resnet18_lr8e-3_wd5e-2_bt256
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenetlt_resnet18_lr8e-3_wd5e-2_bt256.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj199

source activate py3.8_pt1.8.1
python main_lt.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 8e-3 \
	       --weight-decay 5e-2 \
	       -b 256 \
	       -j 64 \
	       --mark workdir_lt/imagenetlt_res34_lr8e-3_wd5e-2_b256 \
	       /mnt/proj198/jqcui/Data/ImageNet
