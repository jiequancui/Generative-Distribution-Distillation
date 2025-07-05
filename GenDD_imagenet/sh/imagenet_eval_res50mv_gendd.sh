#!/bin/bash
#SBATCH --job-name=imagenet_res50mv_gendd_eval
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenet_res50mv_gendd_eval.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj204

source activate py3.8_pt1.8.1

python main_supervised_gendd.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8886' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --teacher_arch resnet50 \
               --num_sampling 5 \
	       -j 64 \
	       -b 512 \
               --mark 'workdir/tmp' \
	       --resume workdir/imagenet_res50mv_gendd/model_best.pth.tar \
               --evaluate \
	       /mnt/proj198/jqcui/Data/ImageNet
