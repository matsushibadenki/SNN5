# ファイルパス: snn_research/architectures/mla_transformer_block.py
# Title: Multi-Level Attention (MLA) Transformer Block
#
# 機能の説明: 複数のレベル（階層）のアテンションを統合する
# Transformerブロック。
#
# 【修正内容 v31.1: ImportError (MultiLevelSpikeDrivenSelfAttention) の修正】
# - health-check 実行時に 'ImportError: cannot import name
#   'MultiLevelSpikeDrivenSelfAttention'' が発生する問題に対処します。
# - (L: 19) 'MultiLevelSpikeDrivenSelfAttention' は、
#   'ComplexSpikeDrivenAttention' (v29で修正) の
#   タイポ（タイプミス）または古い名前であると判断しました。
# - (L: 19) インポート名を 'ComplexSpikeDrivenAttention' に修正しました。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..core.attention import SpikeDrivenSelfAttention
from ..core.neurons import get_neuron_by_name

# --- ▼▼▼ 【!!! 修正 v31.1: ImportError 修正 !!!】 ▼▼▼
# (MultiLevelSpikeDrivenSelfAttention -> ComplexSpikeDrivenAttention)
from ..core.layers.complex_attention import ComplexSpikeDrivenAttention
# --- ▲▲▲ 【!!! 修正 v31.1】 ▲▲▲


class MLATransformerBlock(nn.Module):
    """
    Multi-Level Attention (MLA) Transformer Block
    (中略)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_steps: int,
        neuron_config: Dict[str, Any],
        sdsa_config: Dict[str, Any],
        # (v15) MLA (Complex) Attention Config
        complex_attn_config: Dict[str, Any] 
    ):
        super().__init__()
        self.time_steps = time_steps
        
        # 1. Standard Spike-Driven Self-Attention
        self.sdsa = SpikeDrivenSelfAttention(
            dim=d_model,
            num_heads=nhead,
            time_steps=time_steps,
            neuron_config=neuron_config,
            **sdsa_config
        )
        
        # 2. Multi-Level (Complex) Attention
        # --- ▼▼▼ 【!!! 修正 v31.1: クラス名修正 !!!】 ▼▼▼
        self.complex_attn = ComplexSpikeDrivenAttention(
        # --- ▲▲▲ 【!!! 修正 v31.1】 ▲▲▲
            dim=d_model,
            num_heads=nhead,
            time_steps=time_steps,
            neuron_config=neuron_config,
            **complex_attn_config
        )
        
        # 3. Feedforward Network (FFN)
        self.linear1 = nn.Linear(d_model * 2, dim_feedforward) # (v15: d_model * 2)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # (FFNニューロン)
        neuron_config_ffn1 = neuron_config.copy()
        neuron_config_ffn1['features'] = dim_feedforward
        self.ffn_neuron1 = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_ffn1
        )
        neuron_config_ffn2 = neuron_config.copy()
        neuron_config_ffn2['features'] = d_model
        self.ffn_neuron2 = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_ffn2
        )
        
        # Norm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def set_stateful(self, stateful: bool):
        # (v15)
        if hasattr(self.sdsa, 'set_stateful'):
            self.sdsa.set_stateful(stateful)
        if hasattr(self.complex_attn, 'set_stateful'):
            self.complex_attn.set_stateful(stateful)
        if hasattr(self.ffn_neuron1, 'set_stateful'):
            self.ffn_neuron1.set_stateful(stateful) # type: ignore[attr-defined]
        if hasattr(self.ffn_neuron2, 'set_stateful'):
            self.ffn_neuron2.set_stateful(stateful) # type: ignore[attr-defined]

    def reset(self):
        # (v15)
        if hasattr(self.sdsa, 'reset'):
            self.sdsa.reset()
        if hasattr(self.complex_attn, 'reset'):
            self.complex_attn.reset()
        if hasattr(self.ffn_neuron1, 'reset'):
            self.ffn_neuron1.reset() # type: ignore[attr-defined]
        if hasattr(self.ffn_neuron2, 'reset'):
            self.ffn_neuron2.reset() # type: ignore[attr-defined]


    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """ (v15) (T, B, N, C) を処理 """
        T, B, N, C = src.shape
        outputs = []

        for t in range(T):
            x_t = src[t] # (B, N, C)
            
            # 1. SDSA
            sdsa_out = self.sdsa(x_t)
            
            # 2. Complex Attention
            complex_out = self.complex_attn(x_t)
            
            # 3. Concatenate
            attn_out = torch.cat([sdsa_out, complex_out], dim=-1) # (B, N, C*2)
            
            # (v15) Add & Norm (注: 次元が C -> C*2 に増えているため、
            #      残差接続は FFN の後で行う)
            
            # 4. FFN
            # (B, N, C*2) -> (B, N, D_ff) -> (B, N, C)
            ffn_out = self.linear2(self.dropout(self.ffn_neuron1(self.linear1(attn_out))[0])) # type: ignore[attr-defined]
            ffn_out, _ = self.ffn_neuron2(ffn_out) # type: ignore[attr-defined]

            # 5. Add & Norm (v15: ここで残差接続)
            x_t = self.norm1(x_t + self.dropout(ffn_out))
            
            outputs.append(x_t)
            
        return torch.stack(outputs, dim=0)
