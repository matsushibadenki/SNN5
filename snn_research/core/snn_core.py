# ファイルパス: snn_research/core/snn_core.py
# (リファクタリング版)
#
# Title: SNN Core Factory
# Description:
# - このファイルは、プロジェクト内のSNNモデルアーキテクチャを動的に
#   インスタンス化するための「ファクトリ」クラスである SNNCore のみを定義します。
# - 実際のレイヤーやモデル定義は core/layers/ および core/models/ に分離されました。
# - 修正: 削除対象の旧式モデル (v1, simple) への参照を削除。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, base # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import math
from omegaconf import DictConfig, OmegaConf
from torchvision import models # type: ignore
import logging 

# --- 基底クラスとニューロンのインポート ---
from .base import BaseModel, SNNLayerNorm
from .neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron
)
from snn_research.io.spike_encoder import DifferentiableTTFSEncoder

# --- 分離されたローカルモデルのインポート ---
from .models.predictive_coding_model import BreakthroughSNN
# from .models.spiking_transformer_v1_model import SpikingTransformer_OldTextOnly # 削除
# from .models.simple_snn_model import SimpleSNN # 削除
from .models.hybrid_cnn_snn_model import HybridCnnSnnModel
from .models.spiking_cnn_model import SpikingCNN

# --- 他のSOTAアーキテクチャのインポート ---
from .mamba_core import SpikingMamba
from .trm_core import TinyRecursiveModel
from .sntorch_models import SpikingTransformerSnnTorch 
from snn_research.models.temporal_snn import SimpleRSNN, GatedSNN
from snn_research.architectures.hybrid_transformer import HybridSNNTransformer
from snn_research.architectures.hybrid_attention_transformer import HybridAttentionTransformer
from snn_research.architectures.spiking_rwkv import SpikingRWKV
from snn_research.architectures.sew_resnet import SEWResNet
from snn_research.architectures.tskips_snn import TSkipsSNN
from snn_research.architectures.spiking_ssm import SpikingSSM
from snn_research.architectures.spiking_transformer_v2 import SpikingTransformerV2

logger = logging.getLogger(__name__)

class SNNCore(nn.Module):
    """
    SNNモデルを動的に構築・ラップするファクトリクラス。
    configの architecture_type に基づいて適切なモデルをインスタンス化します。
    (snn_core.pyから他のクラスを分離し、ファクトリ機能のみを残した)
    """
    def __init__(self, config: DictConfig, vocab_size: int, backend: str = "spikingjelly"):
        super(SNNCore, self).__init__()
        
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        model_type: str = self.config.get("architecture_type", "simple")
        self.model: nn.Module
        
        params: Dict[str, Any] = cast(Dict[str, Any], OmegaConf.to_container(self.config, resolve=True))
        params.pop('path', None)
        neuron_config: Dict[str, Any] = params.pop('neuron', {})

        # (v_hpo_fix_value_error, v_hpo_fix_oom_v3 のロジックを維持)
        if model_type == "spiking_transformer":
            # if "num_layers" in params and "num_encoder_layers" not in params:
            #     params["num_encoder_layers"] = params.pop("num_layers") # 削除
            if "d_model" in params and "dim_feedforward" not in params:
                params["dim_feedforward"] = params["d_model"] * 4 # これは正しい
            # if "n_head" in params and "nhead" not in params:
            #     params["nhead"] = params.pop("n_head") # 削除
            if vocab_size == 10: # cifar10
                logger.info("Detected vocab_size=10 (likely CIFAR-10). Overriding ViT params for SpikingTransformerV2.")
                params["img_size"] = 32
                params["patch_size"] = 4
                params["in_channels"] = 3
            else:
                if "img_size" not in params: params["img_size"] = 224
                if "patch_size" not in params: params["patch_size"] = 16
                if "in_channels" not in params: params["in_channels"] = 3

        model_map: Dict[str, Type[BaseModel]]
        if backend == "spikingjelly":
            model_map = {
                # --- ▼ 分離されたローカルモデル ▼ ---
                "predictive_coding": BreakthroughSNN,
                # "spiking_transformer_old": SpikingTransformer_OldTextOnly, # 削除
                # "simple": SimpleSNN, # 削除
                "hybrid_cnn_snn": HybridCnnSnnModel,
                "spiking_cnn": SpikingCNN,
                # --- ▲ 分離されたローカルモデル ▲ ---
                
                # --- ▼ 外部アーキテクチャ ▼ ---
                "spiking_transformer": SpikingTransformerV2, # type: ignore[dict-item]
                "tskips_snn": TSkipsSNN, # type: ignore[dict-item]
                "spiking_ssm": SpikingSSM, # type: ignore[dict-item]
                "temporal_snn": SimpleRSNN,
                "gated_snn": GatedSNN,
                "spiking_mamba": SpikingMamba, 
                "tiny_recursive_model": TinyRecursiveModel, 
                "hybrid_transformer": HybridSNNTransformer,
                "hybrid_attention_transformer": HybridAttentionTransformer,
                "spiking_rwkv": SpikingRWKV,
                "sew_resnet": SEWResNet,
            }
        elif backend == "snntorch":
            model_map = { # type: ignore[assignment]
                "spiking_transformer": SpikingTransformerSnnTorch, 
            }
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if model_type not in model_map:
            # 'simple' が削除されたため、デフォルトを 'predictive_coding' に変更
            if model_type == 'simple':
                logger.warning("Model type 'simple' is deprecated. Defaulting to 'predictive_coding'.")
                model_type = 'predictive_coding'
            else:
                raise ValueError(f"Unknown model type '{model_type}' for backend '{backend}'")
        
        # (パラメータ調整ロジックはすべて維持)
        if model_type in ["temporal_snn", "gated_snn"]:
            params['input_dim'] = config.get('input_dim', 1)
            params['hidden_dim'] = config.get('hidden_dim', 64)
            params['output_dim'] = config.get('output_dim', vocab_size)

        # 'simple' が削除されたため、この分岐は不要になる
        # if 'time_steps' not in params and model_type == 'simple':
        #      params['time_steps'] = config.get('time_steps', 16) 
             
        if model_type in ["spiking_cnn", "sew_resnet"]:
            num_classes_cfg = OmegaConf.select(config, "num_classes", default=None)
            params['num_classes'] = num_classes_cfg if num_classes_cfg is not None else vocab_size
        
        self.model = model_map[model_type](vocab_size=vocab_size, neuron_config=neuron_config, **params)
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # (元の forward ロジックは変更なし)
        model_type: Optional[str] = self.config.get("architecture_type")
        
        input_key: str
        if model_type in ["hybrid_cnn_snn", "spiking_cnn", "sew_resnet"]:
            input_key = "input_images"
        elif model_type in ["tskips_snn", "temporal_snn", "gated_snn"]:
            input_key = "input_sequence"
        elif model_type == "spiking_ssm":
            input_key = "input_ids"
        elif model_type == "spiking_transformer":
            input_key = 'input_ids' # デフォルト (次元チェックで分岐)
        else:
            input_key = 'input_ids'
        
        input_data: Optional[torch.Tensor] = kwargs.get(input_key)
        
        if input_data is None and args and len(args) > 0:
            if isinstance(args[0], torch.Tensor):
                input_data = args[0]

        forward_kwargs: Dict[str, Any] = kwargs.copy()
        if input_key in forward_kwargs:
            del forward_kwargs[input_key] 

        if input_data is None:
            return self.model(**forward_kwargs) # type: ignore[operator]

        if model_type in ["hybrid_cnn_snn", "spiking_cnn", "sew_resnet"]:
            return self.model(input_images=input_data, **forward_kwargs) # type: ignore[operator]
        elif model_type in ["tskips_snn", "temporal_snn", "gated_snn"]:
            return self.model(input_sequence=input_data, **forward_kwargs) # type: ignore[operator]
        else:
             if model_type == "spiking_transformer":
                 if input_data.dim() == 4:
                     return self.model(input_images=input_data, **forward_kwargs) # type: ignore[operator]
                 elif input_data.dim() == 2:
                    return self.model(input_ids=input_data, **forward_kwargs) # type: ignore[operator]
                 else:
                     return self.model(input_ids=input_data, **forward_kwargs) # type: ignore[operator]
             else:
                 return self.model(input_ids=input_data, **forward_kwargs) # type: ignore[operator]
