# ファイルパス: snn_research/core/snn_core.py
# (更新)
#
# Title: SNN Core Models (SNNネイティブAttention改修版)
# (省略...)
# 改善 (v16):
# - doc/SNN開発：基本設計思想.md (セクション3.3, 引用[31]) に基づき、
#   STAttenBlock に「学習可能な遅延」を導入し、TCA問題への対応を強化。
#
# 修正 (v17):
# - mypy [name-defined] [union-attr] [misc] エラーを修正。
#
# 改善 (v19):
# - doc/SNN開発：基本設計思想.md (セクション4.1, 引用[70]) に基づき、
#   AnalogToSpikes モジュールを改修し、DifferentiableTTFSEncoder (学習可能なエンコーダ) を
#   使用できるようにする。
#
# 修正 (v20):
# - mypy [assignment] [no-redef] エラーを修正。
# - AnalogToSpikes.forward 内の DTTFS パスとLIFパスの変数名を分離。
#
# 追加 (v21):
# - SNN5改善レポート (セクション6.2) に基づき、TSkipsSNN をモデルマップに追加。
#
# 修正 (v22):
# - mypy [name-defined] エラー (neuron_type_str) を修正。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, base # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import math
from omegaconf import DictConfig, OmegaConf
from torchvision import models # type: ignore
import logging 

from .base import BaseModel, SNNLayerNorm
# --- ▼ 修正: SNN5改善レポートで追加したニューロンをインポート ▼ ---
from .neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron # SNN5改善レポートによる追加
)
# --- ▲ 修正 ▲ ---
from .mamba_core import SpikingMamba
from .trm_core import TinyRecursiveModel
from .sntorch_models import SpikingTransformerSnnTorch 
from snn_research.io.spike_encoder import DifferentiableTTFSEncoder


# --- ▼ 新しいアーキテクチャのインポート ▼ ---
from snn_research.architectures.hybrid_transformer import HybridSNNTransformer
from snn_research.architectures.hybrid_attention_transformer import HybridAttentionTransformer
from snn_research.architectures.spiking_rwkv import SpikingRWKV
from snn_research.architectures.sew_resnet import SEWResNet
from snn_research.architectures.hybrid_neuron_network import HybridSpikingCNN
# --- ▼ SNN5改善レポートで追加したアーキテクチャをインポート ▼ ---
from snn_research.architectures.tskips_snn import TSkipsSNN
# --- ▲ SNN5改善レポートで追加したアーキテクチャをインポート ▲ ---


logger = logging.getLogger(__name__)


class PredictiveCodingLayer(nn.Module):
    # (変更なし)
    error_mean: torch.Tensor
    error_std: torch.Tensor
    generative_neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron] 
    inference_neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron] 

    def __init__(self, d_model: int, d_state: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.generative_fc = nn.Linear(d_state, d_model)
        self.generative_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron], neuron_class(features=d_model, **self._filter_neuron_params(neuron_class, neuron_params)))
        self.inference_fc = nn.Linear(d_model, d_state)
        self.inference_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron], neuron_class(features=d_state, **self._filter_neuron_params(neuron_class, neuron_params)))
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
        # --- ▼ SNN5改善レポートで追加したニューロンのパラメータ ▼ ---
        elif neuron_class == TC_LIF:
            valid_params = ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
        elif neuron_class == DualThresholdNeuron:
            valid_params = ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
        elif neuron_class == ScaleAndFireNeuron:
            valid_params = ['features', 'num_levels', 'base_threshold']
        # --- ▲ SNN5改善レポートで追加したニューロンのパラメータ ▲ ---

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
        
        combined_mem = gen_mem + inf_mem 
        return updated_state, prediction_error, combined_mem

class MultiLevelSpikeDrivenSelfAttention(nn.Module):
    # (変更なし)
    neuron_out: nn.Module 
    mem_history: List[torch.Tensor]
    neuron_q: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron] 
    neuron_k: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron] 


    def __init__(self, d_model: int, n_head: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any], time_scales: List[int] = [1, 3, 5]):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.time_scales = time_scales
        self.mem_history = []
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model * len(time_scales), d_model)
        
        filtered_params = self._filter_neuron_params(neuron_class, neuron_params)
        self.neuron_q = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron], neuron_class(features=d_model, **filtered_params))
        self.neuron_k = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron], neuron_class(features=d_model, **filtered_params))
        self.neuron_out = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron], neuron_class(features=d_model, **filtered_params))
        
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.01))

    def _filter_neuron_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたニューロンクラスの__init__が受け入れるパラメータのみをフィルタリングする"""
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        elif neuron_class == GLIFNeuron:
            valid_params = ['features', 'base_threshold', 'gate_input_features']
        # --- ▼ SNN5改善レポートで追加したニューロンのパラメータ ▼ ---
        elif neuron_class == TC_LIF:
            valid_params = ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
        elif neuron_class == DualThresholdNeuron:
            valid_params = ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
        elif neuron_class == ScaleAndFireNeuron:
            valid_params = ['features', 'num_levels', 'base_threshold']
        # --- ▲ SNN5改善レポートで追加したニューロンのパラメータ ▲ ---

        
        filtered_params: Dict[str, Any] = {k: v for k, v in neuron_params.items() if k in valid_params}
        return filtered_params

    def _hook_mem(self, module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.mem_history.append(output[1])
    
    def register_mem_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        hooks.append(self.neuron_q.register_forward_hook(self._hook_mem))
        hooks.append(self.neuron_k.register_forward_hook(self._hook_mem))
        hooks.append(self.neuron_out.register_forward_hook(self._hook_mem))
        return hooks

    def clear_mem_history(self) -> None:
        self.mem_history = []

    def _xnor_similarity(self, q_spikes: torch.Tensor, k_spikes: torch.Tensor) -> torch.Tensor:
        q_ext: torch.Tensor = q_spikes.unsqueeze(3)
        k_ext: torch.Tensor = k_spikes.unsqueeze(2)
        xnor_matrix: torch.Tensor = 1.0 - torch.pow(q_ext - k_ext, 2)
        attn_scores: torch.Tensor = xnor_matrix.sum(dim=-1)
        return attn_scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q_raw, _ = self.neuron_q(self.q_proj(x))
        k_raw, _ = self.neuron_k(self.k_proj(x))
        v = self.v_proj(x)

        q_gate = torch.sigmoid(q_raw - self.sparsity_threshold)
        k_gate = torch.sigmoid(k_raw - self.sparsity_threshold)
        q = q_raw * q_gate
        k = k_raw * k_gate

        outputs: List[torch.Tensor] = []
        for scale in self.time_scales:
            if T >= scale and T % scale == 0:
                q_scaled = F.avg_pool1d(q.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                k_scaled = F.avg_pool1d(k.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                v_scaled = F.avg_pool1d(v.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                
                T_scaled = q_scaled.shape[1]

                q_h = q_scaled.view(B, T_scaled, self.n_head, self.d_head).permute(0, 2, 1, 3) 
                k_h = k_scaled.view(B, T_scaled, self.n_head, self.d_head).permute(0, 2, 1, 3) 
                v_h = v_scaled.view(B, T_scaled, self.n_head, self.d_head).permute(0, 2, 1, 3) 
                
                attn_scores_xnor = self._xnor_similarity(q_h, k_h) 
                attn_weights = torch.sigmoid(attn_scores_xnor) 
                attn_output = torch.matmul(attn_weights, v_h) 
                
                attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T_scaled, C)
                attn_output_upsampled = F.interpolate(attn_output.transpose(1, 2), size=T, mode='nearest').transpose(1, 2)
                outputs.append(attn_output_upsampled)

        if not outputs:
             neuron_out_spikes, _ = self.neuron_out(x)
             return cast(torch.Tensor, neuron_out_spikes)

        concatenated_output = torch.cat(outputs, dim=-1)
        final_output = self.out_proj(concatenated_output)
        final_spikes, _ = self.neuron_out(final_output.reshape(B*T, -1))
        return final_spikes.reshape(B, T, C)

class STAttenBlock(nn.Module):
    # (変更なし)
    mem_history: List[torch.Tensor]
    lif1: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron] 
    lif2: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron] 
    lif3: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron] 
    attn: MultiLevelSpikeDrivenSelfAttention
    learned_delays: nn.Parameter

    def __init__(self, d_model: int, n_head: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.norm1 = SNNLayerNorm(d_model)
        self.attn = MultiLevelSpikeDrivenSelfAttention(d_model, n_head, neuron_class, neuron_params)
        self.learned_delays = nn.Parameter(torch.zeros(d_model))

        filtered_params = self._filter_neuron_params(neuron_class, neuron_params)
        self.lif1 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron], neuron_class(features=d_model, **filtered_params))
        self.norm2 = SNNLayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.lif2 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron], neuron_class(features=d_model * 4, **filtered_params))
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.lif3 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron], neuron_class(features=d_model, **filtered_params))
        self.mem_history = []

    def _filter_neuron_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたニューロンクラスの__init__が受け入れるパラメータのみをフィルタリングする"""
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        elif neuron_class == GLIFNeuron:
            valid_params = ['features', 'base_threshold', 'gate_input_features']
        # --- ▼ SNN5改善レポートで追加したニューロンのパラメータ ▼ ---
        elif neuron_class == TC_LIF:
            valid_params = ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
        elif neuron_class == DualThresholdNeuron:
            valid_params = ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
        elif neuron_class == ScaleAndFireNeuron:
            valid_params = ['features', 'num_levels', 'base_threshold']
        # --- ▲ SNN5改善レポートで追加したニューロンのパラメータ ▲ ---
        
        filtered_params: Dict[str, Any] = {k: v for k, v in neuron_params.items() if k in valid_params}
        return filtered_params

    def _hook_mem(self, module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.mem_history.append(output[1])

    def register_mem_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        hooks.extend(self.attn.register_mem_hooks())
        hooks.append(self.lif1.register_forward_hook(self._hook_mem))
        hooks.append(self.lif2.register_forward_hook(self._hook_mem))
        hooks.append(self.lif3.register_forward_hook(self._hook_mem))
        return hooks
    
    def clear_mem_history(self) -> None:
        self.mem_history = []
        self.attn.clear_mem_history()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        attn_out = self.attn(self.norm1(x))
        x_attn = x + attn_out
        x_flat = x_attn.reshape(B * T, D)
        
        x_delayed = x_flat + self.learned_delays
        spike_flat, _ = self.lif1(x_delayed)
        
        x_res = spike_flat.reshape(B, T, D)
        ffn_in = self.norm2(x_res)
        ffn_flat = ffn_in.reshape(B * T, D)
        ffn_hidden, _ = self.lif2(self.fc1(ffn_flat))
        ffn_out_flat = self.fc2(ffn_hidden)
        ffn_out = ffn_out_flat.reshape(B, T, D)
        x_ffn = x_res + ffn_out
        x_ffn_flat = x_ffn.reshape(B * T, D)
        out_flat, _ = self.lif3(x_ffn_flat)
        out = out_flat.reshape(B, T, D)
        return out

class BreakthroughSNN(BaseModel):
    # (変更なし)
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, neuron_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.input_encoder = nn.Linear(d_model, d_model)

        neuron_params: Dict[str, Any] = neuron_config.copy() if neuron_config is not None else {}
        neuron_type_str: str = neuron_params.pop('type', 'lif') 
        neuron_params.pop('num_branches', None)
        neuron_params.pop('branch_features', None)
        
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron]]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        elif neuron_type_str == 'glif':
            neuron_class = GLIFNeuron
            neuron_params['gate_input_features'] = d_model # ゲート入力を d_model に設定
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'base_threshold', 'gate_input_features']
            }
        # --- ▼ SNN5改善レポートで追加したニューロンをサポート ▼ ---
        elif neuron_type_str == 'tc_lif':
            neuron_class = TC_LIF # type: ignore[assignment]
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        # --- ▲ SNN5改善レポートで追加したニューロンをサポート ▲ ---
        else:
            raise ValueError(f"Unknown neuron type for BreakthroughSNN: {neuron_type_str}")

        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, neuron_class, neuron_params) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_state * num_layers, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device: torch.device = input_ids.device
        token_emb: torch.Tensor = self.token_embedding(input_ids)
        embedded_sequence: torch.Tensor = self.input_encoder(token_emb)
        
        inference_neuron = self.pc_layers[0].inference_neuron
        inference_neuron_features: int
        if isinstance(inference_neuron, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF)): # TC_LIFを追加
             inference_neuron_features = inference_neuron.features
        else:
             inference_neuron_features = self.d_state # Fallback
        states: List[torch.Tensor] = [torch.zeros(batch_size, inference_neuron_features, device=device) for _ in range(self.num_layers)]
        
        all_timestep_outputs: List[torch.Tensor] = []
        all_timestep_mems: List[torch.Tensor] = []

        for _ in range(self.time_steps):
            sequence_outputs: List[torch.Tensor] = []
            sequence_mems: List[torch.Tensor] = []
            
            for i in range(seq_len):
                bottom_up_input: torch.Tensor = embedded_sequence[:, i, :]
                layer_mems: List[torch.Tensor] = []
                for j in range(self.num_layers):
                    states[j], error, combined_mem = self.pc_layers[j](bottom_up_input, states[j])
                    bottom_up_input = error
                    layer_mems.append(combined_mem)
                sequence_outputs.append(torch.cat(states, dim=1))
                sequence_mems.append(torch.cat(layer_mems, dim=1)) 

            all_timestep_outputs.append(torch.stack(sequence_outputs, dim=1))
            all_timestep_mems.append(torch.stack(sequence_mems, dim=1))
        
        full_hiddens: torch.Tensor = torch.stack(all_timestep_outputs, dim=2) 
        full_mems: torch.Tensor = torch.stack(all_timestep_mems, dim=2) 
        
        final_hidden_states: torch.Tensor = all_timestep_outputs[-1] 

        output: torch.Tensor
        mem_to_return: torch.Tensor
        
        if output_hidden_states:
             output = final_hidden_states
        elif return_full_hiddens:
             mem_to_return = full_mems if return_full_mems else torch.tensor(0.0, device=device)
             return full_hiddens, torch.tensor(0.0, device=device), mem_to_return
        else:
             output = self.output_projection(final_hidden_states)
        
        total_spikes: float = self.get_total_spikes()
        avg_spikes_val: float = total_spikes / (seq_len * self.time_steps * batch_size) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        
        mem_to_return = full_mems if return_full_mems else torch.tensor(0.0, device=device)
        return output, avg_spikes, mem_to_return

class SpikingTransformer(BaseModel):
    # (変更なし)
    all_mems_history: List[torch.Tensor]
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.d_model = d_model
        self.all_mems_history = []

        # --- ▼ 修正: [name-defined] エラー (neuron_type_str -> neuron_type) ▼ ---
        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron]] 
        
        filtered_params: Dict[str, Any]
        if neuron_type == 'lif':
        # --- ▲ 修正 ▲ ---
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        # --- ▼ 修正: [name-defined] エラー (neuron_type_str -> neuron_type) ▼ ---
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
        # --- ▲ 修正 ▲ ---
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
            cast(base.MemoryModule, block.lif1).set_stateful(True)
            cast(base.MemoryModule, block.lif2).set_stateful(True)
            cast(base.MemoryModule, block.lif3).set_stateful(True)
            cast(base.MemoryModule, block.attn.neuron_q).set_stateful(True)
            cast(base.MemoryModule, block.attn.neuron_k).set_stateful(True)
            cast(base.MemoryModule, block.attn.neuron_out).set_stateful(True)


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
            cast(base.MemoryModule, block.lif1).set_stateful(False)
            cast(base.MemoryModule, block.lif2).set_stateful(False)
            cast(base.MemoryModule, block.lif3).set_stateful(False)
            cast(base.MemoryModule, block.attn.neuron_q).set_stateful(False)
            cast(base.MemoryModule, block.attn.neuron_k).set_stateful(False)
            cast(base.MemoryModule, block.attn.neuron_out).set_stateful(False)


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

class SimpleSNN(BaseModel):
    # (変更なし)
    all_mems_history: List[torch.Tensor]
    lif1: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron] 
    fc1: nn.Linear 

    def __init__(self, vocab_size: int, d_model: int, hidden_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any): 
        super().__init__()
        self.time_steps = time_steps 
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, hidden_size)
        
        # --- ▼ 修正: [name-defined] エラー (neuron_type_str -> neuron_type) ▼ ---
        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron]] 
        
        filtered_params: Dict[str, Any]
        if neuron_type == 'lif':
        # --- ▲ 修正 ▲ ---
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        # --- ▼ 修正: [name-defined] エラー (neuron_type_str -> neuron_type) ▼ ---
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
        # --- ▲ 修正 ▲ ---
            neuron_class = TC_LIF # type: ignore[assignment]
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        else:
             raise ValueError(f"Unknown neuron type for SimpleSNN: {neuron_type}")
        
        self.lif1 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron], neuron_class(features=hidden_size, **filtered_params))
        
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

class AnalogToSpikes(BaseModel): 
    neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, DifferentiableTTFSEncoder]
    all_mems_history: List[torch.Tensor]
    
    def __init__(self, in_features: int, out_features: int, time_steps: int, activation: Type[nn.Module], neuron_config: Dict[str, Any]):
        super().__init__() 
        self.time_steps = time_steps
        self.all_mems_history = []
        self.projection = nn.Linear(in_features, out_features)
        
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, DifferentiableTTFSEncoder]] 
        
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
        # --- ▼ SNN5改善レポートで追加したニューロンをサポート ▼ ---
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
        # --- ▲ SNN5改善レポートで追加したニューロンをサポート ▲ ---
        else:
            raise ValueError(f"Unknown neuron type for AnalogToSpikes: {neuron_type_str}")
        
        # --- ▼ SNN5改善レポートで追加したニューロンをサポート ▼ ---
        self.neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, DifferentiableTTFSEncoder, TC_LIF, DualThresholdNeuron], neuron_class(**filtered_params))
        # --- ▲ SNN5改善レポートで追加したニューロンをサポート ▲ ---
        self.output_act = activation()
    
    def _hook_mem(self, module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.all_mems_history.append(output[1]) # mem
    
    def forward(self, x_analog: torch.Tensor, return_full_mems: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # (B, L, D_in) or (B, D_in)
        x: torch.Tensor = self.projection(x_analog)
        # (B, L, D_out) or (B, D_out)
        x = self.output_act(x)
        
        # --- ▼ 修正: DTTFS (学習可能エンコーダ) の分岐 ▼ ---
        if isinstance(self.neuron, DifferentiableTTFSEncoder):
            # (B, L, D_out) -> (B*L, D_out)
            B, *dims, D_out = x.shape
            x_flat = x.reshape(-1, D_out)
            
            # --- ▼ 修正: [no-redef] エラー解消 ▼ ---
            # (B*L, D_out) -> (B*L, T_steps, D_out)
            dttfs_spikes_stacked = self.neuron(x_flat) 
            
            dttfs_output_shape: Tuple[int, ...] # 
            if x_analog.dim() == 3: # (B, L, D_in)
                dttfs_output_shape = (B, dims[0], self.time_steps, D_out)
            else: # (B, D_in)
                # --- ▼ 修正: [assignment] エラー解消 ▼ ---
                dttfs_output_shape = (B, self.time_steps, D_out) 
                # --- ▲ 修正 ▲ ---
            # --- ▲ 修正 ▲ ---
                
            return dttfs_spikes_stacked.reshape(dttfs_output_shape), None # DTTFSは膜電位を返さない

        # --- 従来のLIF/Izhikevich/GLIF/TC_LIF/DualThreshold (外部T_stepsループ) ---
        x_repeated: torch.Tensor = x.unsqueeze(-2).repeat(1, *([1] * (x_analog.dim() - 1)), self.time_steps, 1)
        
        cast(base.MemoryModule, self.neuron).set_stateful(True)
        functional.reset_net(self.neuron)

        hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self.all_mems_history = []
        if return_full_mems:
            hook = self.neuron.register_forward_hook(self._hook_mem)

        spikes_history: List[torch.Tensor] = []
        
        neuron_features: int = -1
        # --- ▼ SNN5改善レポートで追加したニューロンをサポート ▼ ---
        if isinstance(self.neuron, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)):
            neuron_features = self.neuron.features
        # --- ▲ SNN5改善レポートで追加したニューロンをサポート ▲ ---
        else:
             neuron_features = self.projection.out_features # Fallback
        
        x_time_batched: torch.Tensor = x_repeated.reshape(-1, self.time_steps, neuron_features) 

        for t in range(self.time_steps):
            current_input: torch.Tensor = x_time_batched[:, t, :]
            spike_t, _ = self.neuron(current_input) 
            spikes_history.append(spike_t)
            
        cast(base.MemoryModule, self.neuron).set_stateful(False)
        
        full_mems: Optional[torch.Tensor] = None
        if return_full_mems and hook is not None:
            hook.remove()
            if self.all_mems_history:
                full_mems = torch.stack(self.all_mems_history, dim=1)

        # --- ▼ 修正: [no-redef] エラー解消 (L728) ▼ ---
        spikes_stacked: torch.Tensor = torch.stack(spikes_history, dim=1)
        # --- ▲ 修正 ▲ ---
        
        original_shape: Tuple[int, ...] = x_repeated.shape
        # --- ▼ 修正: [no-redef] / [assignment] エラー解消 (L731, L736) ▼ ---
        output_shape: Tuple[int, ...] 
        
        if x_analog.dim() == 3: # (B, L, D_in)
            output_shape = (original_shape[0], original_shape[1], self.time_steps, neuron_features) 
        else: # (B, D_in)
            output_shape = (original_shape[0], self.time_steps, neuron_features) 
        # --- ▲ 修正 ▲ ---
            
        return spikes_stacked.reshape(output_shape), full_mems
        # --- ▲ 修正 ▲ ---


class HybridCnnSnnModel(BaseModel):
    # (変更なし)
    all_mems_history: List[torch.Tensor]
    def __init__(self, vocab_size: int, time_steps: int, ann_frontend: Dict[str, Any], snn_backend: Dict[str, Any], neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.all_mems_history = []
        
        if ann_frontend['name'] == 'mobilenet_v2':
            weights: Optional[models.MobileNet_V2_Weights] = models.MobileNet_V2_Weights.DEFAULT if ann_frontend.get('pretrained', True) else None
            mobilenet: nn.Module = models.mobilenet_v2(weights=weights)
            self.ann_feature_extractor: nn.Module = mobilenet.features # type: ignore[assignment]
        else:
            raise ValueError(f"Unsupported ANN frontend: {ann_frontend['name']}")
        
        for param in self.ann_feature_extractor.parameters():
             param.requires_grad = True

        self.adapter_a2s = AnalogToSpikes(
            in_features=ann_frontend['output_features'],
            out_features=snn_backend['d_model'],
            time_steps=time_steps,
            activation=nn.ReLU,
            neuron_config=neuron_config # neuron_config をそのまま渡す
        )
        
        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron]] 
        
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
            neuron_params['gate_input_features'] = snn_backend['d_model']
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'base_threshold', 'gate_input_features']
            }
        # --- ▼ SNN5改善レポートで追加したニューロンをサポート ▼ ---
        elif neuron_type == 'tc_lif':
            neuron_class = TC_LIF # type: ignore[assignment]
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        # --- ▲ SNN5改善レポートで追加したニューロンをサポート ▲ ---
        else:
             # dttfs は AnalogToSpikes 内部でのみ使用
             raise ValueError(f"Unknown neuron type for HybridCnnSnnModel backend: {neuron_type}")
        
        self.snn_backend = nn.ModuleList([
            STAttenBlock(snn_backend['d_model'], snn_backend['n_head'], neuron_class, filtered_params)
            for _ in range(snn_backend['num_layers'])
        ])

        self.output_projection = nn.Linear(snn_backend['d_model'], vocab_size)
        self._init_weights()
        
    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = input_images.shape
        device: torch.device = input_images.device
        functional.reset_net(self) 

        ann_features: torch.Tensor = self.ann_feature_extractor(input_images)
        ann_features = ann_features.mean([2, 3]) 
        
        snn_input_spikes: torch.Tensor
        adapter_mems: Optional[torch.Tensor]
        snn_input_spikes, adapter_mems = self.adapter_a2s(ann_features, return_full_mems=return_full_mems) 
        
        self.all_mems_history = []
        if return_full_mems and adapter_mems is not None:
            self.all_mems_history.append(adapter_mems.unsqueeze(1)) 
            
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        if return_full_mems:
            for layer_module in self.snn_backend:
                block = cast(STAttenBlock, layer_module)
                block.clear_mem_history()
                hooks.extend(block.register_mem_hooks())


        x: torch.Tensor = snn_input_spikes
        functional.reset_net(self.snn_backend) 
        
        full_hiddens_list: List[torch.Tensor] = []
        
        for layer_module in self.snn_backend:
             block = cast(STAttenBlock, layer_module)
             cast(base.MemoryModule, block.lif1).set_stateful(True)
             cast(base.MemoryModule, block.lif2).set_stateful(True)
             cast(base.MemoryModule, block.lif3).set_stateful(True)
             cast(base.MemoryModule, block.attn.neuron_q).set_stateful(True)
             cast(base.MemoryModule, block.attn.neuron_k).set_stateful(True)
             cast(base.MemoryModule, block.attn.neuron_out).set_stateful(True)

             x = layer_module(x) # type: ignore[operator]
             full_hiddens_list.append(x)
             
             cast(base.MemoryModule, block.lif1).set_stateful(False)
             cast(base.MemoryModule, block.lif2).set_stateful(False)
             cast(base.MemoryModule, block.lif3).set_stateful(False)
             cast(base.MemoryModule, block.attn.neuron_q).set_stateful(False)
             cast(base.MemoryModule, block.attn.neuron_k).set_stateful(False)
             cast(base.MemoryModule, block.attn.neuron_out).set_stateful(False)

        full_mems: torch.Tensor
        if return_full_mems:
            for layer_idx, layer_module in enumerate(self.snn_backend):
                block = cast(STAttenBlock, layer_module)
                num_neurons_in_block = 6
                block_mems = block.mem_history
                lif3_mems: List[torch.Tensor] = [block_mems[t*num_neurons_in_block + 5] for t in range(self.time_steps) if (t*num_neurons_in_block + 5) < len(block_mems)]
                if len(lif3_mems) == self.time_steps:
                    lif3_mems_stacked: torch.Tensor = torch.stack(lif3_mems, dim=1).unsqueeze(1) 
                    self.all_mems_history.append(lif3_mems_stacked)

            for hook in hooks: hook.remove()
            
            if self.all_mems_history:
                 full_mems_cat: torch.Tensor = torch.cat(self.all_mems_history, dim=4) 
                 full_mems = full_mems_cat
            else:
                 full_mems = torch.zeros(B, 1, self.time_steps, 1, device=device) 
        else:
            full_mems = torch.tensor(0.0, device=device)

        full_hiddens: torch.Tensor = torch.stack(full_hiddens_list, dim=1) 
            
        final_features: torch.Tensor = x.mean(dim=1)
        
        if return_full_hiddens:
             return full_hiddens, torch.tensor(0.0, device=device), full_mems
             
        logits: torch.Tensor = self.output_projection(final_features)
        
        total_spikes: float = self.get_total_spikes()
        avg_spikes_val: float = total_spikes / (B * self.time_steps) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        
        return logits, avg_spikes, full_mems

class SpikingCNN(BaseModel):
    # (変更なし)
    all_mems_history: List[torch.Tensor]
    
    def __init__(self, vocab_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        num_classes: int = vocab_size
        self.time_steps = time_steps
        self.all_mems_history = []
        
        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron]] 
        
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
        # --- ▼ SNN5改善レポートで追加したニューロンをサポート ▼ ---
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
        # --- ▲ SNN5改善レポートで追加したニューロンをサポート ▲ ---
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


class SNNCore(nn.Module):
    # (変更なし)
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

        
        model_map: Dict[str, Type[BaseModel]]
        if backend == "spikingjelly":
            model_map = {
                "predictive_coding": BreakthroughSNN,
                "spiking_transformer": SpikingTransformer,
                "spiking_mamba": SpikingMamba, 
                "tiny_recursive_model": TinyRecursiveModel, 
                "simple": SimpleSNN,
                "hybrid_cnn_snn": HybridCnnSnnModel,
                "spiking_cnn": SpikingCNN,
                "hybrid_transformer": HybridSNNTransformer,
                "hybrid_attention_transformer": HybridAttentionTransformer,
                "spiking_rwkv": SpikingRWKV,
                "sew_resnet": SEWResNet,
                # --- ▼ SNN5改善レポートで追加したアーキテクチャを登録 ▼ ---
                "tskips_snn": TSkipsSNN, # type: ignore[dict-item]
                # --- ▲ SNN5改善レポートで追加したアーキテクチャを登録 ▲ ---
            }
        elif backend == "snntorch":
            model_map = { # type: ignore[assignment]
                "spiking_transformer": SpikingTransformerSnnTorch, 
            }
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if model_type not in model_map:
            raise ValueError(f"Unknown model type '{model_type}' for backend '{backend}'")
        
        if 'time_steps' not in params and model_type == 'simple':
             params['time_steps'] = config.get('time_steps', 16) 
             
        if model_type in ["spiking_cnn", "sew_resnet"]:
            num_classes_cfg = OmegaConf.select(config, "num_classes", default=None)
            if num_classes_cfg is not None:
                params['num_classes'] = num_classes_cfg
            else:
                params['num_classes'] = vocab_size
        
        self.model = model_map[model_type](vocab_size=vocab_size, neuron_config=neuron_config, **params)
        

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        model_type: Optional[str] = self.config.get("architecture_type")
        
        # --- ▼ SNN5改善レポートで追加したアーキテクチャの入力キー ▼ ---
        input_key: str
        if model_type in ["hybrid_cnn_snn", "spiking_cnn", "sew_resnet"]:
            input_key = "input_images"
        elif model_type == "tskips_snn":
            input_key = "input_sequence"
        # --- ▲ SNN5改善レポートで追加したアーキテクチャの入力キー ▲ ---
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

        # --- ▼ SNN5改善レポートで追加したアーキテクチャの入力キー ▼ ---
        if model_type in ["hybrid_cnn_snn", "spiking_cnn", "sew_resnet"]:
            return self.model(input_images=input_data, **forward_kwargs) # type: ignore[operator]
        elif model_type == "tskips_snn":
            return self.model(input_sequence=input_data, **forward_kwargs) # type: ignore[operator]
        # --- ▲ SNN5改善レポートで追加したアーキテクチャの入力キー ▲ ---
        else:
            return self.model(input_ids=input_data, **forward_kwargs) # type: ignore[operator]
