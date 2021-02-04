# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import random
import configargparse
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils

class Net(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hidden_layer_num):
        super(Net, self).__init__()

        self.first_layer = nn.Linear(feature_dim, hidden_dim)

        self.layers = nn.ModuleList()

        for i in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.predict = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.first_layer(x))

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.predict(x)

        return x


class LatencyPredictor(object):
    def __init__(self, feature_norm, lat_norm, ckpt_path, feature_dim, hidden_dim, hidden_layer_num):
        self.feature_norm = np.array(feature_norm)
        self.lat_norm = lat_norm
        self.ckpt_path = ckpt_path
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_layer_num = hidden_layer_num
        self.model = Net(self.feature_dim, self.hidden_dim, self.hidden_layer_num)

    def load_ckpt(self):
        self.model.load_state_dict(torch.load(self.ckpt_path))

    def predict_lat(self, config):
        with torch.no_grad():
            features = utils.get_config_features(config)
            features_norm = np.array(features) / self.feature_norm

            prediction = self.model(torch.Tensor(features_norm)).item() * self.lat_norm

        return prediction

if __name__=='__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--configs', required=True, is_config_file=True)
    parser.add_argument('--feature-norm', type=float, nargs='+', default=[640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2], help='normalizing factor for each feature')
    parser.add_argument('--lat-norm', type=float, default=200, help='normalizing factor for latency')
    parser.add_argument('--ckpt-path', type=str, required=True, help='path to save latency predictor weights')
    parser.add_argument('--hidden-dim', type=int, default=400, help='hidden dimension of FC layers in latency predictor')
    parser.add_argument('--hidden-layer-num', type=int, default=3, help='number of FC layers')
    parser.add_argument('--feature-dim', type=int, default=10, help='dimension of feature vector')

    args = parser.parse_args()
    print(args)

    predictor = LatencyPredictor(feature_norm=args.feature_norm, lat_norm=args.lat_norm, ckpt_path=args.ckpt_path, 
                                 feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, hidden_layer_num=args.hidden_layer_num)
    predictor.load_ckpt()
    config_example = {
        'encoder': {
            'encoder_embed_dim': 512,
            'encoder_layer_num': 12,
            'encoder_ffn_embed_dim': [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
            'encoder_self_attention_heads': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        },
        'decoder': {
            'decoder_embed_dim': 512,
            'decoder_layer_num': 3,
            'decoder_ffn_embed_dim': [1024, 1024, 1024],
            'decoder_self_attention_heads': [4, 4, 4],
            'decoder_ende_attention_heads': [4, 4, 4],
            'decoder_arbitrary_ende_attn':  [-1, -1, -1]
        }
    }

    predict = predictor.predict_lat(config_example)
    print(f'Model config: {config_example}')
    print(f'Predicted latency: %.2fms' % predict)
