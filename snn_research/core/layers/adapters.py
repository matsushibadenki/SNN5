# ファイルパス: snn_research/core/layers/adapters.py
# (snn_core.pyから分離)
#
# Title: ANN-SNN アダプタレイヤー
# Description:
# - HybridCnnSnnModelなどで使用される、アナログ値とスパイク時系列を
#   相互変換するためのアダプタレイヤー。
# - 学習可能なTTFSエンコーダ (DifferentiableTTFSEncoder) も含む。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import math

from ..base import BaseModel, SNNLayerNorm
from ..neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron
)
from snn_research.io.spike_encoder import DifferentiableTTFSEncoder

from spikingjelly.activation_based import functional, base as sj_base # type: ignore[import-untyped]

# AnalogToSpikes は BaseModel を継承
class AnalogToSpikes(BaseModel): 
    """
    アナログ値をスパイク時系列に変換するアダプタ。
    DifferentiableTTFSEncoder (DTTFS) のロジックも含む。
    (snn_core.pyから分離)
    """
    neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, DifferentiableTTFSEncoder, TC_LIF, DualThresholdNeuron] # 修正: TC_LIF, DualThresholdNeuron追加
    all_mems_history: List[torch.Tensor]
    
    def __init__(self, in_features: int, out_features: int, time_steps: int, activation: Type[nn.Module], neuron_config: Dict[str, Any]):
        super().__init__() 
        self.time_steps = time_steps
        self.all_mems_history = []
        self.projection = nn.Linear(in_features, out_features)
        
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, DifferentiableTTFSEncoder, TC_LIF, DualThresholdNeuron]] # 修正: TC_LIF, DualThresholdNeuron追加
        
        filtered_params: Dict[str, Any]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        elif neuron_type_str == 'glif':
            neuron_class = GLIFNeuron
            neuron_params['gate_input_features'] = out_features
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'base_threshold', 'gate_input_features']
            }
        elif neuron_type_str == 'dttfs': # 設計思想 4.1
            neuron_class = DifferentiableTTFSEncoder
            neuron_params['num_neurons'] = out_features
            neuron_params['duration'] = time_steps
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['num_neurons', 'duration', 'initial_sensitivity']
            }
        elif neuron_type_str == 'tc_lif':
            neuron_class = TC_LIF # type: ignore[assignment]
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        elif neuron_type_str == 'dual_threshold':
            neuron_class = DualThresholdNeuron # type: ignore[assignment]
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
            }
        else:
            raise ValueError(f"Unknown neuron type for AnalogToSpikes: {neuron_type_str}")
        
        self.neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, DifferentiableTTFSEncoder, TC_LIF, DualThresholdNeuron], neuron_class(**filtered_params))
        self.output_act = activation()
    
    def _hook_mem(self, module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.all_mems_history.append(output[1]) # mem
    
    def forward(self, x_analog: torch.Tensor, return_full_mems: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # (B, L, D_in) or (B, D_in)
        x: torch.Tensor = self.projection(x_analog)
        # (B, L, D_out) or (B, D_out)
        x = self.output_act(x)
        
        if isinstance(self.neuron, DifferentiableTTFSEncoder):
            # (B, L, D_out) -> (B*L, D_out)
            B, *dims, D_out = x.shape
            x_flat = x.reshape(-1, D_out)
            
            # (B*L, D_out) -> (B*L, T_steps, D_out)
            dttfs_spikes_stacked = self.neuron(x_flat) 
            
            dttfs_output_shape: Tuple[int, ...]
            if x_analog.dim() == 3: # (B, L, D_in)
                dttfs_output_shape = (B, dims[0], self.time_steps, D_out)
            else: # (B, D_in)
                dttfs_output_shape = (B, self.time_steps, D_out) 
                
            return dttfs_spikes_stacked.reshape(dttfs_output_shape), None # DTTFSは膜電位を返さない

        # --- 従来のLIF/Izhikevich/GLIF/TC_LIF/DualThreshold (外部T_stepsループ) ---
        x_repeated: torch.Tensor = x.unsqueeze(-2).repeat(1, *([1] * (x_analog.dim() - 1)), self.time_steps, 1)
        
        # 修正: sj_base.MemoryModuleを継承していることを確認
        if isinstance(self.neuron, sj_base.MemoryModule):
            cast(sj_base.MemoryModule, self.neuron).set_stateful(True)
        functional.reset_net(self.neuron)

        hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self.all_mems_history = []
        if return_full_mems:
            hook = self.neuron.register_forward_hook(self._hook_mem)

        spikes_history: List[torch.Tensor] = []
        
        neuron_features: int = -1
        if isinstance(self.neuron, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)):
            neuron_features = self.neuron.features
        else:
             neuron_features = self.projection.out_features # Fallback
        
        x_time_batched: torch.Tensor = x_repeated.reshape(-1, self.time_steps, neuron_features) 

        for t in range(self.time_steps):
            current_input: torch.Tensor = x_time_batched[:, t, :]
            spike_t, _ = self.neuron(current_input) 
            spikes_history.append(spike_t)
            
        if isinstance(self.neuron, sj_base.MemoryModule):
            cast(sj_base.MemoryModule, self.neuron).set_stateful(False)
        
        full_mems: Optional[torch.Tensor] = None
        if return_full_mems and hook is not None:
            hook.remove()
            if self.all_mems_history:
                full_mems = torch.stack(self.all_mems_history, dim=1)

        spikes_stacked: torch.Tensor = torch.stack(spikes_history, dim=1)
        
        original_shape: Tuple[int, ...] = x_repeated.shape
        output_shape: Tuple[int, ...] 
        
        if x_analog.dim() == 3: # (B, L, D_in)
            output_shape = (original_shape[0], original_shape[1], self.time_steps, neuron_features) 
        else: # (B, D_in)
            output_shape = (original_shape[0], self.time_steps, neuron_features) 
            
        return spikes_stacked.reshape(output_shape), full_mems
