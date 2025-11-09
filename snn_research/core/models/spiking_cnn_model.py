# ファイルパス: snn_research/core/models/spiking_cnn_model.py
# (snn_core.pyから分離)
#
# Title: Spiking CNN Model
# Description:
# - CIFAR-10などの画像分類タスク用の基本的なSpiking CNNモデル。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union

from ..base import BaseModel
from ..neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)

from spikingjelly.activation_based import functional # type: ignore[import-untyped]

class SpikingCNN(BaseModel):
    """
    画像分類用のSpiking CNNモデル。
    (snn_core.pyから分離)
    """
    all_mems_history: List[torch.Tensor]
    
    def __init__(self, vocab_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        num_classes: int = vocab_size
        self.time_steps = time_steps
        self.all_mems_history = []
        
        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron]] # 修正: TC_LIF, DualThresholdNeuron追加
        
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
            neuron_params['gate_input_features'] = None 
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
        elif neuron_type == 'dual_threshold':
            neuron_class = DualThresholdNeuron # type: ignore[assignment]
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
            }
        else:
             raise ValueError(f"Unknown neuron type for SpikingCNN: {neuron_type}")

        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            neuron_class(features=16, **filtered_params), # [0]
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            neuron_class(features=32, **filtered_params), # [1]
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), # 224x224入力の場合
            neuron_class(features=128, **filtered_params), # [2]
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = input_images.shape
        device: torch.device = input_images.device
        functional.reset_net(self)
        
        output_voltages: List[torch.Tensor] = []
        full_hiddens_list: List[torch.Tensor] = [] 
        self.all_mems_history = []
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        neuron_layers: List[nn.Module] = []
        if return_full_mems:
            def _hook_mem(module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
                self.all_mems_history.append(output[1]) # mem
            
            for module in self.features:
                if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    hooks.append(module.register_forward_hook(_hook_mem))
                    neuron_layers.append(module)
            for module in self.classifier:
                 if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    hooks.append(module.register_forward_hook(_hook_mem))
                    neuron_layers.append(module)

        for _ in range(self.time_steps):
            x: torch.Tensor = input_images
            
            hidden_repr_t: Optional[torch.Tensor] = None 
            
            for features_layer in self.features: 
                if isinstance(features_layer, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    B_c, C_c, H_c, W_c = x.shape
                    x_reshaped: torch.Tensor = x.permute(0, 2, 3, 1).reshape(-1, C_c)
                    spikes, _ = features_layer(x_reshaped) # type: ignore[operator]
                    x = spikes.view(B_c, H_c, W_c, C_c).permute(0, 3, 1, 2)
                else:
                    x = features_layer(x) # type: ignore[operator]
                
                if isinstance(x, tuple):
                    x = x[0]
            
            hidden_repr_t = x.mean(dim=[2, 3]) 
            full_hiddens_list.append(hidden_repr_t) 

            for i, classifier_layer in enumerate(self.classifier): 
                
                if isinstance(classifier_layer, nn.Flatten):
                    x = classifier_layer(x) # type: ignore[operator]
                    continue
                elif isinstance(classifier_layer, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    spikes, _ = classifier_layer(x) # type: ignore[operator]
                    x = spikes
                elif isinstance(classifier_layer, nn.Linear):
                    if not isinstance(x, torch.Tensor):
                         x = cast(torch.Tensor, x)
                    x = classifier_layer(x) # type: ignore[operator]
                if isinstance(x, tuple):
                    x = x[0]

            output_voltages.append(x) 
        
        full_mems: torch.Tensor
        if return_full_mems:
            for hook in hooks: hook.remove()
            if self.all_mems_history:
                num_layers: int = len(neuron_layers)
                mems_by_time: List[List[torch.Tensor]] = [[] for _ in range(self.time_steps)]
                for i, mem in enumerate(self.all_mems_history):
                    t: int = i % self.time_steps
                    mems_by_time[t].append(mem.view(B, -1)) 
                
                mems_stacked_time: List[torch.Tensor] = [torch.cat(mems_t, dim=1) for mems_t in mems_by_time]
                full_mems_stacked: torch.Tensor = torch.stack(mems_stacked_time, dim=1)
                full_mems = full_mems_stacked.unsqueeze(1)
            else:
                full_mems = torch.zeros(B, 1, self.time_steps, 1, device=device) 
        else:
            full_mems = torch.tensor(0.0, device=device)

        
        full_hiddens_stacked: torch.Tensor = torch.stack(full_hiddens_list, dim=1) 
        full_hiddens: torch.Tensor = full_hiddens_stacked.unsqueeze(1) 

        if return_full_hiddens:
             return full_hiddens, torch.tensor(0.0, device=device), full_mems

        final_logits: torch.Tensor = torch.stack(output_voltages, dim=0).mean(dim=0)
        
        total_spikes: float = self.get_total_spikes()
        avg_spikes_val: float = total_spikes / (B * self.time_steps) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)

        return final_logits, avg_spikes, full_mems
