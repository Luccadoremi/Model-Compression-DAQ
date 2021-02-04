train_url=$1
ARCH=$2
config=$3
sub_config=$4
GPU=$5
update_freq=$6

cd /cache/code_dir
ls -l
pip install numpy==1.16.0
pip install --editable .
#python setup.py build develop --user
  
# train with 8 GPU cards

CUDA_VISIBLE_DEVICES=${GPU} python train-${ARCH}.py \
         --configs=$config \
         --sub-configs=$sub_config \
         --update-freq=${update_freq} \
         --save-dir=../model_dir	&

echo "##### perodically cp back checkpoint #####"

python run_cp_checkpoint.py --train_url $train_url

wait
