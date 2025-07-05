#!/bin/bash
#SBATCH --job-name=imagenetlt_gendd_resnext101res50
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenetlt_gendd_resnext101res50.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj197

source activate py3.8_pt1.8.1


python main_gendd_lt.py -a resnet50 \
	       --dist-url 'tcp://127.0.0.1:8889' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
               --cos \
	       --weight-decay 2e-1 \
	       --teacher_arch resnext101_32x4d \
	       --teacher_model workdir_lt/imagenetlt_sgd_resnext101/model_best.pth.tar \
	       --smooth 0.0 \
	       --epochs 100 \
	       -j 64 \
	       -b 256 \
               --mark 'workdir_lt/imagenetlt_resnext101res50_gendd_unsupervised' \
	       /mnt/proj198/jqcui/Data/ImageNet
