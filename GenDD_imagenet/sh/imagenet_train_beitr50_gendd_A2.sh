#!/bin/bash
#SBATCH --job-name=imagenet_beitres50_gendd_A2
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenet_beitres50_gendd_A2.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj204

source activate py3.8_pt1.8.1
python main_gendd_pretrain.py -a resnet50 \
	       --dist-url 'tcp://127.0.0.1:8888' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
               --teacher_arch beitv2_base_patch16_224 \
	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_base_patch16_224_pt1k_ft21kto1k_new.pth \
	       --mixup 0.1 \
	       --cutmix 1.0 \
	       --smoothing 0.0 \
               --aug_type 'rand' \
               -j 64 \
	       -b 512 \
	       --epochs 20 \
	       --mark 'workdir/imagenet_beitres50_pretrain_20e_A2' \
	       /mnt/proj198/jqcui/Data/ImageNet


python main_supervised_gendd.py -a resnet50 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --cos \
	       --weight-decay 2e-2 \
               --teacher_arch beitv2_base_patch16_224 \
	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_base_patch16_224_pt1k_ft21kto1k_new.pth \
	       --smooth 0.9 \
	       --beta 0.0 \
	       --epochs 300 \
	       --mixup 0.1 \
	       --cutmix 1.0 \
	       --smoothing 0.0 \
               --aug_type 'rand' \
	       -j 64 \
	       -b 512 \
               --reload workdir/imagenet_beitres50_pretrain_20e_A2/checkpoint.pth.tar \
	       --mark 'workdir/imagenet_beitres50_gendd_A2' \
	       /mnt/proj198/jqcui/Data/ImageNet

