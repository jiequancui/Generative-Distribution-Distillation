#!/bin/bash
#SBATCH --job-name=resnet34_lr1e-3_wd5e-2_cos
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=resnet34_lr1e-3_wd5e-2.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj192

source activate py3.8_pt1.8.1
python main_wd_cosine.py -a resnet34 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 1e-3 \
	       --weight-decay 5e-2 \
	       -b 512 \
	       -j 64 \
	       --mark 'workdir/resnet34_wdcos'\
	       /mnt/proj198/jqcui/Data/ImageNet
