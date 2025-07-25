#!/bin/bash
#SBATCH --job-name=imagenetlt_sgd_resnext101
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenetlt_sgd_resnext101.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj199

source activate py3.8_pt1.8.1
python main_lt_sgd.py -a resnext101_32x4d \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 0.1 \
	       --weight-decay 5e-4 \
	       -b 256 \
	       -j 64 \
	       --mark workdir_lt/imagenetlt_sgd_resnext101 \
	       /mnt/proj198/jqcui/Data/ImageNet
