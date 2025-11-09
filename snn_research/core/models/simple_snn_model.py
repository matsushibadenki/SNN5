# ファイルパス: snn_research/core/models/simple_snn_model.py
# (snn_core.pyから分離)
#
# Title: Simple SNN Model
# Description:
# - デバッグやスモークテスト用のシンプルなSNNモデル。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union

from ..base import BaseModel
from ..neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF
)

from spikingjelly.activation_based import functional # type: ignore[import-untyped]

class SimpleSNN(BaseModel):
    """
    デバッグ用のシンプルなSNN。
    (snn_core.pyから分離)
    """
    all_mems_history: List[torch.Tensor]
    lif1: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF] # 修正: TC_LIF追加
    fc1: nn.Linear 

    def __init__(self, vocab_size: int, d_model: int, hidden_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any): 
        super().__init__()
        self.time_steps = time_steps 
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, hidden_size)
        
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
            neuron_params['gate_input_features'] = hidden_size
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
             raise ValueError(f"Unknown neuron type for SimpleSNN: {neuron_type}")
        
        self.lif1 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF], neuron_class(features=hidden_size, **filtered_params))
        
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self._init_weights()
        self.all_mems_history = []

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device 
        x = self.embedding(input_ids)
        outputs: List[torch.Tensor] = []
        functional.reset_net(self)
        
        full_hiddens_list: List[torch.Tensor] = [] 
        self.all_mems_history = []
        hook: Optional[torch.utils.hooks.RemovableHandle] = None
        if return_full_mems:
            def _hook_mem(module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
                self.all_mems_history.append(output[1]) # mem
            hook = self.lif1.register_forward_hook(_hook_mem)
        
        x_avg = x.mean(dim=1) 
        
        x_current = self.fc1(x_avg) # (B, hidden_size)
        
        for _ in range(self.time_steps):
            out, _ = self.lif1(x_current) # (B, hidden_size)
            full_hiddens_list.append(out) 
            out_logits = self.fc2(out)
            outputs.append(out_logits)
            
        logits = torch.stack(outputs, dim=1) 
        logits = logits.mean(dim=1) 
        
        full_mems: torch.Tensor
        if return_full_mems and hook is not None:
            hook.remove()
            if self.all_mems_history:
                full_mems_stacked = torch.stack(self.all_mems_history, dim=1) 
                full_mems = full_mems_stacked.unsqueeze(1) 
            else:
                full_mems = torch.zeros(B, 1, self.time_steps, self.fc1.out_features, device=device) 
        else:
            full_mems = torch.tensor(0.0, device=device)

        
        full_hiddens_stacked = torch.stack(full_hiddens_list, dim=1) 
        full_hiddens = full_hiddens_stacked.unsqueeze(1) 
        
        if return_full_hiddens:
             return full_hiddens, torch.tensor(0.0, device=device), full_mems
             
        avg_spikes_val: float = self.get_total_spikes() / (B * self.time_steps) if return_spikes else 0.0 
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        return logits, avg_spikes, full_mems
