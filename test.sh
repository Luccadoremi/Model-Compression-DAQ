#!/bin/bash -x

##################################
#  RUNNING WITH THIS PARAMETERS  #
##################################
ARCH=${1:-"hat"}
TYPE=${2:-"no_noise"}
QUANTCONFIG=${3:-""}
GPU=${4:-"0,1"}
DATA=${5:-"iwslt14.de-en"}
METRICS=${6:-"normal"}
subset=${7:-"test"}


CHK=checkpoints
CHK_FULL=${CHK}/${DATA}/${ARCH}-${TYPE}

SUB_TFMR=configs/${DATA}/subtransformer

# if we are readin from quantized model so we need to read last checkpoint
if test -z ${QUANTCONFIG}
then
  checkpoints_path=${CHK_FULL}/checkpoint_best.pt
else
  checkpoints_path=${CHK_FULL}/checkpoint_last.pt
fi

configs=${CHK_FULL}/model.yml

output_path=$(dirname -- "$checkpoints_path")
out_name=$(basename -- "$configs")

mkdir -p $output_path/exp
PATH=/mnt/sdj/turkeymt/talha/envs/hat-dev/bin:$PATH

CUDA_VISIBLE_DEVICES=${GPU} python generate.py \
        --data data/binary/wmt16_en_de  \
        --path "$checkpoints_path" \
        --gen-subset $subset \
        --beam 5 \
        --batch-size 128 \
        --remove-bpe \
        --configs=$configs \
        --quantization_config_path=${QUANTCONFIG} \
        > $output_path/exp/${out_name}_${subset}_gen.out

GEN=$output_path/exp/${out_name}_${subset}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

# get normal BLEU or SacreBLEU score
if [ ${METRICS} = "normal" ]
then
  echo "Evaluate Normal BLEU score!"
  grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
  grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
  python score.py --sys $SYS --ref $REF
elif [ ${METRICS} = "sacre" ]
then
  echo "Evaluate SacreBLEU score!"
  grep ^H $GEN | cut -f3- > $SYS.pre
  grep ^T $GEN | cut -f2- > $REF.pre
  sacremoses detokenize < $SYS.pre > $SYS
  sacremoses detokenize < $REF.pre > $REF
  python score.py --sys $SYS --ref $REF --sacrebleu
fi
