#!/bin/bash
#SBATCH --job-name=resnet18_lr1e-4_wd5e-2
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=resnet18_lr1e-4_wd5e-2.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -x proj77

source activate py3.8_pt1.8.1
python main.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 1e-4 \
	       --weight-decay 5e-2 \
	       /mnt/proj198/jqcui/Data/ImageNet
