#!/bin/bash
#SBATCH --job-name=imagenet_res50mv_gendd
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenet_res50mv_gendd.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj204

source activate py3.8_pt1.8.1
python main_gendd_pretrain.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8888' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
	       --teacher_arch resnet50 \
               -j 64 \
	       -b 512 \
	       --epochs 20 \
	       --mark 'workdir/imagenet_res50mv_pretrain_20e' \
	       /mnt/proj198/jqcui/Data/ImageNet


python main_supervised_gendd.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8886' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --cos \
	       --weight-decay 2e-2 \
	       --teacher_arch resnet50 \
	       --smooth 0.9 \
	       --epochs 100 \
	       -j 64 \
	       -b 512 \
               --mark 'workdir/imagenet_res50mv_gendd' \
               --reload workdir/imagenet_res50mv_pretrain_20e/checkpoint.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
