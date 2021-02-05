# Model-Compression-DAQ
This repo implements a new method called DAQ (Divide-and-Quantize) which essentially divides big weight matrices into flexible chunks and quantizes them separately. 


## Usage

### Installation
To install from source and develop locally:

```bash
git clone https://github.com/Luccadoremi/Model-Compression-DAQ.git
cd hat
pip install --editable .
```

### Training

#### 1. Train a SuperTransformer (HAT [[paper]](https://arxiv.org/abs/2005.14187))
The SuperTransformer is a supernet that contains many SubTransformers with weight-sharing.
By default, we train WMT tasks on 8 GPUs. Please adjust `--update-freq` according to GPU numbers (`128/x` for x GPUs). Note that for IWSLT, we only train on one GPU with `--update-freq=1`. 
```bash
python train-our.py --configs=configs/[task_name]/supertransformer/[search_space].yml
# for example
python train-our.py --configs=configs/wmt14.en-de/supertransformer/space0.yml
# another example
CUDA_VISIBLE_DEVICES=0,1,2,3 python train-our.py --configs=configs/wmt14.en-fr/supertransformer/space0.yml --update-freq=32
```
In the `--configs` file, SuperTransformer model architecture, SubTransformer search space and training settings are specified.

#### 3. Train a Searched SubTransformer (Training with Quantization Noise for Extreme Model Compression [[paper]](https://arxiv.org/abs/2004.07320))
For details please check the script.

```bash
# run with default arguments 
./train.sh

# for example this will run a subtransformer training with quantization noise
./train.sh our quant_noise

# this will quantized all weights for details check corresponding yml files
./train.sh our post_quant-quant_noise-n5

# to provide model.yml for a dataset, train.sh can be run like following
# ./train.sh <ARCH> <COMMON.YML-TYPE> <GPUs> <DATASET> <MODEL.YML>
./train.sh our post_quant-quant_noise-n5 0,1 iwslt14.de-en HAT_iwslt14deen_titanxp@168.8ms_bleu@34.8.yml
```
#### Test BLEU (SacreBLEU) score:
For details please check the script.

```bash
# run with default arguments
./test.sh

# Calculate BLEU score for non-quantized model
./test.sh our quant_noise

# Calculate BLEU score for a quantized model (you need to provide quantization config path)
./test.sh our post_quant-quant_noise-n5 configs/iwslt14.de-en/subtransformer/pq-quantization-n5.yml
```

### Dependencies
* Python >= 3.6
* [PyTorch](http://pytorch.org/) >= 1.0.0
* configargparse >= 0.14
* New model training requires NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* sklearn

## Roadmap


- Use 4 bit to encode assignments using more buckets (for now its 8 bits)
- Shared centroids accross the layers
- 1D weight resampling https://github.com/adefossez/julius/

## Licence

This repository is released under the MIT license. See [LICENSE](./LICENSE) for more information.

## Acknowledgements

[fairseq](https://github.com/pytorch/fairseq), [HAT](https://github.com/mit-han-lab/hardware-aware-transformers)
=======
