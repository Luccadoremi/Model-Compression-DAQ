#!/bin/bash -x

##################################
#  RUNNING WITH THIS PARAMETERS  #
##################################
ARCH=${1:-"our"}
TYPE=${2:-"quant_noise"}
GPU=${3:-"0,1"}
DATA=${4:-"iwslt14.de-en"}
MDL=${5:-"HAT_iwslt14deen_titanxp@168.8ms_bleu@34.8.yml"}
UPFRQ=${6:-"1"}
DOWN=${7:-"1.0"}

CHK=checkpoints
LOG=logs

if [ ${DOWN} = "1.0" ]
then
  CHK_FULL=${CHK}/${DATA}/${ARCH}-${TYPE}
else
  CHK_FULL=${CHK}/${DATA}/${ARCH}-${TYPE}-down_rate_${DOWN}
fi

SUB_TFMR=configs/${DATA}/subtransformer

# removing old checkpoints
#rm -rf ${CHK_FULL}
mkdir -p ${CHK_FULL}
mkdir -p $LOG
# copy model's and common yaml to checkpoint directory
cp -f ${SUB_TFMR}/${MDL} ${CHK_FULL}/model.yml
cp -f ${SUB_TFMR}/common-${TYPE}.yml ${CHK_FULL}
PATH=/mnt/sdj/turkeymt/talha/envs/hat-dev/bin:$PATH
#echo $PATH
#echo $(which /usr/bin/env python)
CUDA_VISIBLE_DEVICES=${GPU} /usr/bin/env python train-${ARCH}.py \
         --configs=${SUB_TFMR}/${MDL} \
         --sub-configs=${SUB_TFMR}/common-${TYPE}.yml \
         --update-freq=${UPFRQ} \
         --save-dir=${CHK_FULL} \
         --task "translation_sampling" \
         --downsample-ratio ${DOWN}