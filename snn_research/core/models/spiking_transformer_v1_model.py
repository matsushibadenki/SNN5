# ファイルパス: snn_research/core/models/spiking_transformer_v1_model.py
# (snn_core.pyから分離)
#
# Title: Spiking Transformer (v1 / OldTextOnly)
# Description:
# - STAttenBlock (MultiLevelSpikeDrivenSelfAttention) を使用した初期の
#   テキスト専用Spiking Transformerモデル。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import math

from ..base import BaseModel, SNNLayerNorm
from ..layers.complex_attention import STAttenBlock
from ..neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF
)
from spikingjelly.activation_based import base as sj_base

class SpikingTransformer_OldTextOnly(BaseModel):
    """
    STAttenBlockを使用した旧Spiking Transformer（テキスト専用）。
    (snn_core.pyから分離)
    """
    all_mems_history: List[torch.Tensor]
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.d_model = d_model
        self.all_mems_history = []

        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF]] # 修正: TC_LIF追加
        
        filtered_params: Dict[str, Any]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        elif neuron_type == 'glif':
            neuron_class = GLIFNeuron
            neuron_params['gate_input_features'] = d_model
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'base_threshold', 'gate_input_features']
            }
        elif neuron_type == 'tc_lif':
            neuron_class = TC_LIF # type: ignore[assignment]
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        else:
             raise ValueError(f"Unknown neuron type for SpikingTransformer: {neuron_type}")
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        self.layers = nn.ModuleList([STAttenBlock(d_model, n_head, neuron_class, filtered_params) for _ in range(num_layers)])
        self.final_norm = SNNLayerNorm(d_model) 
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device: torch.device = input_ids.device
        x: torch.Tensor = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        if return_full_mems:
            self.all_mems_history = []
            for layer_module in self.layers:
                block = cast(STAttenBlock, layer_module)
                block.clear_mem_history()
                hooks.extend(block.register_mem_hooks())

        for layer_module in self.layers:
            block = cast(STAttenBlock, layer_module)
            cast(sj_base.MemoryModule, block.lif1).set_stateful(True)
            cast(sj_base.MemoryModule, block.lif2).set_stateful(True)
            cast(sj_base.MemoryModule, block.lif3).set_stateful(True)
            cast(sj_base.MemoryModule, block.attn.neuron_q).set_stateful(True)
            cast(sj_base.MemoryModule, block.attn.neuron_k).set_stateful(True)
            cast(sj_base.MemoryModule, block.attn.neuron_out).set_stateful(True)


        full_hiddens_list: List[torch.Tensor] = []
        for _ in range(self.time_steps):
            for layer_module in self.layers:
                layer: STAttenBlock = cast(STAttenBlock, layer_module)
                x = layer(x)
            
            full_hiddens_list.append(x)
        
        full_hiddens: torch.Tensor = torch.stack(full_hiddens_list, dim=2) 

        full_mems: torch.Tensor
        if return_full_mems:
            layer_mems_by_time: List[List[torch.Tensor]] = [[] for _ in range(self.time_steps)]
            
            for layer_idx, layer_module in enumerate(self.layers):
                block = cast(STAttenBlock, layer_module)
                num_neurons_in_block = 6 
                block_mems = block.mem_history
                lif3_mems: List[torch.Tensor] = [block_mems[t*num_neurons_in_block + 5] for t in range(self.time_steps) if (t*num_neurons_in_block + 5) < len(block_mems)]
                
                if len(lif3_mems) == self.time_steps:
                    lif3_mems_stacked: torch.Tensor = torch.stack(lif3_mems, dim=1) 
                    lif3_mems_stacked = lif3_mems_stacked.view(batch_size, seq_len, self.time_steps, self.d_model)
                    self.all_mems_history.append(lif3_mems_stacked)
                
            for hook in hooks: hook.remove()
            
            if self.all_mems_history:
                 full_mems = torch.stack(self.all_mems_history, dim=0) 
                 full_mems = full_mems.permute(1, 2, 3, 0, 4).reshape(batch_size, seq_len, self.time_steps, -1)
            else:
                 full_mems = torch.zeros_like(full_hiddens) 
        else:
            device = input_ids.device 
            full_mems = torch.tensor(0.0, device=device) 
        

        for layer_module in self.layers:
            block = cast(STAttenBlock, layer_module)
            cast(sj_base.MemoryModule, block.lif1).set_stateful(False)
            cast(sj_base.MemoryModule, block.lif2).set_stateful(False)
            cast(sj_base.MemoryModule, block.lif3).set_stateful(False)
            cast(sj_base.MemoryModule, block.attn.neuron_q).set_stateful(False)
            cast(sj_base.MemoryModule, block.attn.neuron_k).set_stateful(False)
            cast(sj_base.MemoryModule, block.attn.neuron_out).set_stateful(False)


        x_normalized = self.final_norm(x)
        
        output: torch.Tensor
        if output_hidden_states:
            output = x_normalized
        elif return_full_hiddens:
             return full_hiddens, torch.tensor(0.0, device=device), full_mems
        else:
            output = self.output_projection(x_normalized)
        
        total_spikes: float = self.get_total_spikes()
        avg_spikes_val: float = total_spikes / (seq_len * self.time_steps * batch_size) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        
        return output, avg_spikes, full_mems