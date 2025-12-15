#!/bin/bash
#SBATCH --job-name=cc3m_res50mv_unsupervised_kd
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=cc3m_res50mv_unsupervised_kd.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj204

source activate py3.8_pt1.8.1
python main_unsupervised_kd.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8886' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 0.2 \
	       --weight-decay 1e-4 \
	       --teacher_arch resnet50 \
	       --warmup_epochs 10 \
	       --epochs 100 \
	       -j 64 \
	       -b 512 \
               --mark 'workdir/cc3m_res50mv_kd' \
	       /mnt/proj198/jqcui/Data/ImageNet
