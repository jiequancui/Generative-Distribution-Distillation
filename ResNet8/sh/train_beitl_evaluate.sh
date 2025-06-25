#!/bin/bash
#SBATCH --job-name=beit_lr2e-3_wd5e-2_w08_mix_2048_200e
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=beit_lr2e-3_wd5e-2_w08_mix_2048_200e.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj192

source activate py3.8_pt1.8.1
python main_distill_stage23_cosine.py -a resnet50 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 5e-2 \
	       --teacher_arch beitv2_large_patch16_224 \
	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_large_patch16_224_pt1k_ft21kto1k_new.pth \
	       --smooth 0.8 \
	       --beta 1.0 \
	       --diffusion_batch_mul 50 \
	       --diffloss_w 1024 \
	       --warmup_epochs 10 \
	       --epochs 200 \
	       -j 64 \
	       -b 512 \
	       --mark 'workdir/tmp' \
	       --reload workdir/beitlarger50_stage24_lr2e-3_wd2e-2_w09_mix0_1024_300e_crop06_rand_mixup01_cutmix10_A1_pretrain20e_resume/model_best.pth.tar \
	       --evaluate \
	       /mnt/proj198/jqcui/Data/ImageNet
