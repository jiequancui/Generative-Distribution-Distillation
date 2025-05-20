#!/bin/bash
#SBATCH --job-name=res34res18_lr5e-4_wd5e-2_stage2_1024_mix
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr5e-4_wd5e-2_stage2_1024_mix.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -x proj77,proj194

source activate py3.8_pt1.8.1
python main_distill_stage22_cosine.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8888' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 5e-4 \
	       --weight-decay 5e-2 \
	       --teacher_arch resnet34 \
	       --mark 'workdir/tmp' \
	       --reload workdir/res34res18_lr2e-3_wd2e-2_stage22_1024_mix_w09_token64_drop02_bt512_cos_mul5_adam_beta_min095/model_best.pth.tar \
               --evaluate \
	       /mnt/proj198/jqcui/Data/ImageNet


#workdir/res34res18_lr2e-3_wd2e-2_stage22_1024_mix_w07_bt512_cos_mul5_adambeta0999/model_best.pth.tar
#workdir/res34res18_lr2e-3_wd2e-2_stage22_1024_mix_w07_bt2048_cos_mul5_adambeta0999/model_best.pth.tar \

