#!/bin/bash
#SBATCH --job-name=imagenet_res34res18_gendd_linear
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenet_res34res18_gendd_linear.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj77

source activate py3.8_pt1.8.1


python main_supervised_gendd_linear.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8889' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 30 \
	       --cos \
	       --weight-decay 0 \
	       --teacher_arch resnet34 \
	       --epochs 100 \
	       --warmup_epochs 10 \
	       -j 32 \
	       -b 512 \
               --mark 'workdir/imagenet_res34res18_gendd_linear' \
	       --reload workdir/imagenet_res34res18_gendd/model_best.pth.tar \
	       --finetune_classifier \
	       --alpha 0.5 \
	       /mnt/proj198/jqcui/Data/ImageNet
