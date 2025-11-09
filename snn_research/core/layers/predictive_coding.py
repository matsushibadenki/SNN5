# ファイルパス: snn_research/core/layers/predictive_coding.py
# (snn_core.pyから分離)
#
# Title: 予測符号化レイヤー
# Description:
# - BreakthroughSNNモデルの中核となる、予測符号化を実行するレイヤー。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import logging

from ..base import SNNLayerNorm
from ..neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron
)
from snn_research.io.spike_encoder import DifferentiableTTFSEncoder # _filter_neuron_params用

logger = logging.getLogger(__name__)

class PredictiveCodingLayer(nn.Module):
    """
    BreakthroughSNNで使用される予測符号化レイヤー。
    (snn_core.pyから分離)
    """
    error_mean: torch.Tensor
    error_std: torch.Tensor
    generative_neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF] # 修正: TC_LIF追加
    inference_neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF] # 修正: TC_LIF追加

    def __init__(self, d_model: int, d_state: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.generative_fc = nn.Linear(d_state, d_model)
        # 修正: _filter_neuron_params に渡すニューロンクラスの型ヒントを拡張
        self.generative_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF], neuron_class(features=d_model, **self._filter_neuron_params(neuron_class, neuron_params)))
        self.inference_fc = nn.Linear(d_model, d_state)
        self.inference_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF], neuron_class(features=d_state, **self._filter_neuron_params(neuron_class, neuron_params)))
        self.norm_state = SNNLayerNorm(d_state)
        self.norm_error = SNNLayerNorm(d_model)
        self.error_scale = nn.Parameter(torch.ones(1))
        
        self.register_buffer('error_mean', torch.zeros(1))
        self.register_buffer('error_std', torch.ones(1))
        self.error_momentum = 0.9

    def _filter_neuron_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたニューロンクラスの__init__が受け入れるパラメータのみをフィルタリングする"""
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        elif neuron_class == GLIFNeuron:
            valid_params = ['features', 'base_threshold', 'gate_input_features']
        elif neuron_class == DifferentiableTTFSEncoder:
            valid_params = ['num_neurons', 'duration', 'initial_sensitivity']
        elif neuron_class == TC_LIF:
            valid_params = ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
        elif neuron_class == DualThresholdNeuron:
            valid_params = ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
        elif neuron_class == ScaleAndFireNeuron:
            valid_params = ['features', 'num_levels', 'base_threshold']

        filtered_params: Dict[str, Any] = {k: v for k, v in neuron_params.items() if k in valid_params}
        return filtered_params

    def forward(self, bottom_up_input: torch.Tensor, top_down_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction, gen_mem = self.generative_neuron(self.generative_fc(self.norm_state(top_down_state)))
        raw_error = bottom_up_input - prediction
        
        if self.training:
            with torch.no_grad():
                batch_mean = raw_error.mean()
                batch_std = raw_error.std() + 1e-5
                self.error_mean = self.error_momentum * self.error_mean + (1 - self.error_momentum) * batch_mean
                self.error_std = self.error_momentum * self.error_std + (1 - self.error_momentum) * batch_std
        
        normalized_error = (raw_error - self.error_mean) / self.error_std
        prediction_error = normalized_error * self.error_scale
        
        state_update, inf_mem = self.inference_neuron(self.inference_fc(self.norm_error(prediction_error)))
        updated_state = top_down_state * 0.9 + state_update * 0.1
        
        # 結合 (torch.cat) に修正する
        combined_mem = torch.cat((gen_mem, inf_mem), dim=1) # (B, d_model + d_state)
        
        return updated_state, prediction_error, combined_mem