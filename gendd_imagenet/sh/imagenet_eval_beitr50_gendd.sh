#!/bin/bash
#SBATCH --job-name=imagenet_beitres50_gendd_eval
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenet_beitres50_gendd_eval.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj200

source activate py3.8_pt1.8.1

python main_supervised_gendd.py -a resnet50 \
	       --dist-url 'tcp://127.0.0.1:8882' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --cos \
	       --weight-decay 2e-2 \
               --teacher_arch beitv2_base_patch16_224 \
	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_base_patch16_224_pt1k_ft21kto1k_new.pth \
	       -j 64 \
	       -b 512 \
	       --mark 'workdir/tmp' \
	       --resume workdir/imagenet_beitres50_gendd_A2/model_best.pth.tar \
	       --evaluate \
               --num_sampling 5 \
	       /mnt/proj198/jqcui/Data/ImageNet

