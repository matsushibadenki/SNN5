# ファイルパス: snn_research/core/sntorch_models.py
# (新規作成)
# Title: snnTorchベースのSNNモデル実装
# Description:
# - snn_4_ann_parity_plan.md に基づく、学習バックエンドの多様化の一環。
# - snnTorchライブラリを使用してSpikingTransformerモデルを実装する。
# - mypyエラー[import-untyped]を解消するため、type: ignoreを追加。
#
# 修正(v2): forwardメソッドが output_hidden_states と return_full_hiddens を
#           正しく処理するように修正。
#
# 修正(v4): 構文エラー解消のため、末尾の余分な '}' を削除。
#
# 修正 (v5): SyntaxError: 末尾の余分な '}' を削除。
#
# 修正 (v6): 構文エラー解消のため、SpikingTransformerSnnTorch クラスを閉じる '}' を末尾に追加。

import torch
import torch.nn as nn
import snntorch as snn  # type: ignore
from snntorch import surrogate # type: ignore
from typing import Tuple, Dict, Any, List, Optional

from .base import BaseModel, SNNLayerNorm

class STAttenBlockSnnTorch(nn.Module):
    """snnTorchを使用したSpiking Transformerブロック。"""
    def __init__(self, d_model: int, n_head: int, neuron_params: Dict[str, Any]):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        
        beta = neuron_params.get('beta', 0.95)
        grad = surrogate.fast_sigmoid() # () を追加

        self.norm1 = SNNLayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.lif_q = snn.Leaky(beta=beta, surrogate_gradient=grad, init_hidden=True)
        self.lif_k = snn.Leaky(beta=beta, surrogate_gradient=grad, init_hidden=True)

        self.norm2 = SNNLayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.lif1 = snn.Leaky(beta=beta, surrogate_gradient=grad, init_hidden=True)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.lif2 = snn.Leaky(beta=beta, surrogate_gradient=grad, init_hidden=True)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_norm1 = self.norm1(x)
        
        q = self.q_proj(x_norm1)
        k = self.k_proj(x_norm1)
        v = self.v_proj(x_norm1)

        q_spikes, _ = self.lif_q(q)
        k_spikes, _ = self.lif_k(k)

        q_spikes = q_spikes.view(B, T, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k_spikes = k_spikes.view(B, T, self.n_head, self.d_head).permute(0, 2, 3, 1)
        v = v.view(B, T, self.n_head, self.d_head).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q_spikes, k_spikes) / (self.d_head ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        
        x_res = x + attn_output
        x_norm2 = self.norm2(x_res)
        
        ffn_out, _ = self.lif1(self.fc1(x_norm2))
        ffn_out = self.fc2(ffn_out)
        ffn_out, _ = self.lif2(x_res + ffn_out)
        
        return ffn_out

class SpikingTransformerSnnTorch(BaseModel):
    """snnTorchをバックエンドとして使用するSpiking Transformerモデル。"""
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        neuron_params = {'beta': neuron_config.get('beta', 0.95)}
        self.layers = nn.ModuleList([
            STAttenBlockSnnTorch(d_model, n_head, neuron_params) for _ in range(num_layers)
        ])
        
        self.final_norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :T, :]
        
        total_spikes = 0
        all_time_outputs: List[torch.Tensor] = []
        
        for layer in self.modules():
            if isinstance(layer, snn.Leaky):
                layer.reset_hidden()

        for _ in range(self.time_steps):
            x_step = x
            for layer in self.layers:
                x_step = layer(x_step)
            
            with torch.no_grad():
                total_spikes += (x_step > 0).float().sum().item()
            
            all_time_outputs.append(x_step)

        x_avg: torch.Tensor = torch.stack(all_time_outputs, dim=0).mean(dim=0)
        
        x_normalized: torch.Tensor = self.final_norm(x_avg)
        
        output: torch.Tensor
        if output_hidden_states:
            output = x_normalized # (B, S, D)
        elif return_full_hiddens:
            output = torch.stack(all_time_outputs, dim=2)
        else:
            output = self.output_projection(x_normalized)
        
        avg_spikes_val = total_spikes / (B * T * self.time_steps) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=input_ids.device)
        
        try:
            final_layer_lif = list(self.layers[-1].modules())[-1]
            if isinstance(final_layer_lif, snn.Leaky):
                 mem = final_layer_lif.mem
            else:
                 mem = torch.tensor(0.0, device=input_ids.device)
        except Exception:
             mem = torch.tensor(0.0, device=input_ids.device)
        
        return output, avg_spikes, mem

}
