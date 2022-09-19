"""ViT Transformers."""
from collections.abc import Mapping
import logging
import math
import os
import time
from typing import Optional, Union
import numpy as np
import requests
import torch
from torch import nn
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTSelfAttention, ViTSelfOutput, ViTIntermediate, ViTOutput
from .. import ModuleShardConfig
from . import TransformerShard, TransformerShardData


logger = logging.getLogger(__name__)

_WEIGHTS_URLS = {
    'google/vit-base-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz',
    'google/vit-large-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz',
    'google/vit-huge-patch14-224-in21k': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
}


def _forward_kernel(layer, x, skip, kernel_id):
    if kernel_id == 1:
        x = layer[0](x)
        x = layer[1](x)[0]
    elif kernel_id == 2:
        x = layer[0](x, skip)
        x += skip
        skip = x
    elif kernel_id == 3:
        x = layer[0](x)
        x = layer[1](x)
    else:
        x = layer[0](x, skip)
        skip = x
    return x, skip


class ViTTransformerShard(TransformerShard):
    """ViT transformer shard."""

    def __init__(self, shard_config: ModuleShardConfig, model_name: str,
                 model_weights: Union[str, Mapping]):
        super().__init__(shard_config, model_name, model_weights)
        if self.model_name == 'google/vit-huge-patch14-224-in21k':
            # This ViT-Huge model doesn't include classification, so we have to set this ourselves
            # NOTE: not setting 'id2label' or 'label2id'
            self.config.num_labels = 21843
        self.embeddings = None
        self.layernorm = None
        self.classifier = None

        logger.debug(">>>> Model name: %s", model_name)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", self.model_weights)
            with np.load(self.model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

        logger.info("======= Finish Build ViTTransformerShard%d ==========", self.shard_config.stage)

    def _build_shard(self, weights):
        ## first Shard
        if self.shard_config.is_first:
            self.embeddings = ViTEmbeddings(self.config)
            logger.debug(">>>> Load embeddings layer for the first shard")
            self._load_layer_weights(weights, load_first=True)

        current_layer_idx = self.shard_config.layer_start

        ## partial model layer
        if self.shard_config.layer_start %4 != 1 or (self.shard_config.layer_start+3 > self.shard_config.layer_end):
            for i in range(self.shard_config.layer_start, min(self.shard_config.layer_end, math.ceil(self.shard_config.layer_start/4)*4)+1):
                logger.debug("    Load the %d-th operation for %d-th layer", i%4, math.ceil(i/4)-1)
                layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1)
                self.first_ops.append(layer)
            current_layer_idx = min(self.shard_config.layer_end+1, math.ceil(self.shard_config.layer_start/4)*4+1)

        ## whole model layers
        while current_layer_idx + 3 <= self.shard_config.layer_end:
            layer = ViTLayer(self.config)
            self._load_layer_weights(weights, model_layer_id=math.ceil(current_layer_idx/4)-1, model_layer=layer)
            self.model_layers.append(layer)
            logger.debug(">>>> Load the %d-th layer", math.ceil(current_layer_idx/4)-1)
            current_layer_idx += 4

        ## partial model layer after whole model layers
        for i in range(current_layer_idx, self.shard_config.layer_end+1):
            logger.debug("    Load the %d-th operation for %d-th layer", i%4, math.ceil(i/4)-1)
            layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1)
            self.last_ops.append(layer)

        ## last Shard
        if self.shard_config.is_last:
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            logger.debug(">>>> Load layernorm for the last shard")
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels) if self.config.num_labels > 0 else nn.Identity()
            logger.debug(">>>> Load classifier for the last shard")
            self._load_layer_weights(weights, load_last=True)

    def _build_kernel(self, weights, kernel_id, model_layer_id):
        layers = nn.ModuleList()
        if kernel_id == 1:
            layers.append(nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps))
            layers.append(ViTSelfAttention(self.config))
        elif kernel_id == 2:
            layers.append(ViTSelfOutput(self.config))
        elif kernel_id == 3:
            layers.append(nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps))
            layers.append( ViTIntermediate(self.config))
        else:
            layers.append(ViTOutput(self.config))
        self._load_layer_weights(weights, model_layer_id=model_layer_id, model_layer=layers, kernel_id=kernel_id)
        return layers

    def _load_layer_weights(self, weights, model_layer_id=0, model_layer=None,
                            load_first=False, load_last=False, kernel_id=None):
        if load_first:
            with torch.no_grad():
                self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["Transformer/posembed_input/pos_embedding"])))
                conv_weight = weights["embedding/kernel"]
                # O, I, J, K = conv_weight.shape
                # conv_weight = conv_weight.reshape(K,J,O,I)
                conv_weight = conv_weight.transpose([3, 2, 0, 1])
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))

        if load_last:
            with torch.no_grad():
                self.layernorm.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
                self.layernorm.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))
                self.classifier.weight.copy_(torch.from_numpy(np.transpose(weights["head/kernel"])))
                self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))

        if model_layer is not None:
            root = f"Transformer/encoderblock_{model_layer_id}/"
            attention_q = root + "MultiHeadDotProductAttention_1/query/"
            attention_k = root + "MultiHeadDotProductAttention_1/key/"
            attention_v = root + "MultiHeadDotProductAttention_1/value/"
            attention_out = root + "MultiHeadDotProductAttention_1/out/"
            fc_0 = root + "MlpBlock_3/Dense_0/"
            fc_1 = root + "MlpBlock_3/Dense_1/"
            attention_norm = root + "LayerNorm_0/"
            mlp_norm = root + "LayerNorm_2/"
            hidden_size = self.config.hidden_size
            with torch.no_grad():
                if kernel_id is None:
                    query_weight = torch.from_numpy(weights[attention_q + "kernel"]).view(hidden_size, hidden_size).t()
                    key_weight = torch.from_numpy(weights[attention_k + "kernel"]).view(hidden_size, hidden_size).t()
                    value_weight = torch.from_numpy(weights[attention_v + "kernel"]).view(hidden_size, hidden_size).t()
                    out_weight = torch.from_numpy(weights[attention_out + "kernel"]).view(hidden_size, hidden_size).t()

                    query_bias = torch.from_numpy(weights[attention_q + "bias"]).view(-1)
                    key_bias = torch.from_numpy(weights[attention_k + "bias"]).view(-1)
                    value_bias = torch.from_numpy(weights[attention_v + "bias"]).view(-1)
                    out_bias = torch.from_numpy(weights[attention_out + "bias"]).view(-1)

                    model_layer.attention.attention.query.weight.copy_(query_weight)
                    model_layer.attention.attention.key.weight.copy_(key_weight)
                    model_layer.attention.attention.value.weight.copy_(value_weight)
                    model_layer.attention.output.dense.weight.copy_(out_weight)

                    model_layer.attention.attention.query.bias.copy_(query_bias)
                    model_layer.attention.attention.key.bias.copy_(key_bias)
                    model_layer.attention.attention.value.bias.copy_(value_bias)
                    model_layer.attention.output.dense.bias.copy_(out_bias)

                    mlp_weight_0 = torch.from_numpy(weights[fc_0 + "kernel"]).t()
                    mlp_weight_1 = torch.from_numpy(weights[fc_1 + "kernel"]).t()
                    mlp_bias_0 = torch.from_numpy(weights[fc_0 + "bias"]).t()
                    mlp_bias_1 = torch.from_numpy(weights[fc_1 + "bias"]).t()

                    model_layer.intermediate.dense.weight.copy_(mlp_weight_0)
                    model_layer.intermediate.dense.bias.copy_(mlp_bias_0)
                    model_layer.output.dense.weight.copy_(mlp_weight_1)
                    model_layer.output.dense.bias.copy_(mlp_bias_1)

                    model_layer.layernorm_before.weight.copy_(torch.from_numpy(weights[attention_norm + "scale"]))
                    model_layer.layernorm_before.bias.copy_(torch.from_numpy(weights[attention_norm + "bias"]))
                    model_layer.layernorm_after.weight.copy_(torch.from_numpy(weights[mlp_norm + "scale"]))
                    model_layer.layernorm_after.bias.copy_(torch.from_numpy(weights[mlp_norm + "bias"]))
                elif kernel_id == 1:
                    query_weight = torch.from_numpy(weights[attention_q + "kernel"]).view(hidden_size, hidden_size).t()
                    key_weight = torch.from_numpy(weights[attention_k + "kernel"]).view(hidden_size, hidden_size).t()
                    value_weight = torch.from_numpy(weights[attention_v + "kernel"]).view(hidden_size, hidden_size).t()

                    query_bias = torch.from_numpy(weights[attention_q + "bias"]).view(-1)
                    key_bias = torch.from_numpy(weights[attention_k + "bias"]).view(-1)
                    value_bias = torch.from_numpy(weights[attention_v + "bias"]).view(-1)

                    model_layer[0].weight.copy_(torch.from_numpy(weights[attention_norm + "scale"]))
                    model_layer[0].bias.copy_(torch.from_numpy(weights[attention_norm + "bias"]))
                    model_layer[1].query.weight.copy_(query_weight)
                    model_layer[1].key.weight.copy_(key_weight)
                    model_layer[1].value.weight.copy_(value_weight)

                    model_layer[1].query.bias.copy_(query_bias)
                    model_layer[1].key.bias.copy_(key_bias)
                    model_layer[1].value.bias.copy_(value_bias)
                elif kernel_id == 2:
                    out_weight = torch.from_numpy(weights[attention_out + "kernel"]).view(hidden_size, hidden_size).t()
                    out_bias = torch.from_numpy(weights[attention_out + "bias"]).view(-1)
                    model_layer[0].dense.weight.copy_(out_weight)
                    model_layer[0].dense.bias.copy_(out_bias)
                elif kernel_id == 3:
                    model_layer[0].weight.copy_(torch.from_numpy(weights[mlp_norm + "scale"]))
                    model_layer[0].bias.copy_(torch.from_numpy(weights[mlp_norm + "bias"]))
                    mlp_weight_0 = torch.from_numpy(weights[fc_0 + "kernel"]).t()
                    mlp_bias_0 = torch.from_numpy(weights[fc_0 + "bias"]).t()
                    model_layer[1].dense.weight.copy_(mlp_weight_0)
                    model_layer[1].dense.bias.copy_(mlp_bias_0)
                elif kernel_id == 0:
                    mlp_weight_1 = torch.from_numpy(weights[fc_1 + "kernel"]).t()
                    mlp_bias_1 = torch.from_numpy(weights[fc_1 + "bias"]).t()
                    model_layer[0].dense.weight.copy_(mlp_weight_1)
                    model_layer[0].dense.bias.copy_(mlp_bias_1)

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        start = time.time()
        x, skip = TransformerShard.parse_forward_data(data)

        if self.shard_config.is_first:
            x = self.embeddings(x)
            skip = x

        for i, op in enumerate(self.first_ops):
            x, skip = _forward_kernel(op, x, skip, (self.shard_config.layer_start+i)%4)

        for i, layer in enumerate(self.model_layers):
            x = layer(x)[0]
            skip = x

        for i, op in enumerate(self.last_ops):
            # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with _load_layer_weights()
            x, skip = _forward_kernel(op, x, skip, (i+1)%4)

        if self.shard_config.is_last:
            x = self.layernorm(x)
            x = self.classifier(x[:, 0, :])
        end = time.time()

        logger.info("Shard%d: computed microbatch in: %f sec", self.shard_config.stage, end - start)

        if self.shard_config.layer_end % 2 == 0:
            return x
        return x, skip

    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
                     timeout_sec: Optional[float]=None) -> None:
        """Save the model weights file."""
        if url is None:
            url = _WEIGHTS_URLS[model_name]
        logger.info('Downloading model: %s: %s', model_name, url)
        req = requests.get(url, stream=True, timeout=timeout_sec)
        req.raise_for_status()
        with open(model_file, 'wb') as file:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())
