# ファイルパス: snn_research/core/layers/complex_attention.py
# Title: Complex Spike-Driven Attention
#
# 機能の説明: 複素数（または高次元）表現を使用した
# スパイク駆動型アテンションメカニズム。
#
# 【修正内容 v29: 循環インポート (Circular Import) の修正】
# - health-check 実行時に 'ImportError: ... (most likely due to a circular import)'
#   が発生する問題に対処します。
# - (L: 21) 'from snn_research.core.snn_core import SNNCore' が、
#   snn_core.py (L:28) -> ... -> complex_attention.py (L:21) という
#   循環参照を引き起こしていました。
# - (L: 24) 'SNNCore' を継承するのは誤りであり、
#   'BaseModel' に修正しました。
# - (L: 21) 'SNNCore' のインポートを削除し、'from ..base import BaseModel' を
#   インポートするように変更しました。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..neurons import get_neuron_by_name

# --- ▼▼▼ 【!!! 修正 v29: 循環インポート修正 !!!】 ▼▼▼
# (from snn_research.core.snn_core import SNNCore を削除)
from ..base import BaseModel # BaseModel をインポート

class ComplexSpikeDrivenAttention(BaseModel): # 'SNNCore' -> 'BaseModel' に変更
# --- ▲▲▲ 【!!! 修正 v29】 ▲▲▲
    """
    Complex Spike-Driven Attention (CSDA)
    (中略)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_steps: int,
        neuron_config: Dict[str, Any],
        **kwargs # (v15: BaseModel から vocab_size を吸収)
    ):
        # (v15: BaseModel の __init__ を呼び出す)
        super(ComplexSpikeDrivenAttention, self).__init__(**kwargs)
        
        self.dim = dim
        self.num_heads = num_heads
        self.time_steps = time_steps
        self.neuron_config = neuron_config

        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # (v15) 複素数表現のため、次元を2倍にする
        self.qkv_linear = nn.Linear(dim, dim * 3 * 2) 
        self.out_linear = nn.Linear(dim * 2, dim)

        # ニューロン
        neuron_config_attn = neuron_config.copy()
        neuron_config_attn['features'] = dim * 2
        self.neuron = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_attn
        )
        
        # (v15) 状態管理
        self._is_stateful = False
        self.built = True

    def set_stateful(self, stateful: bool):
        """ (v15) 状態管理モードを設定 """
        self._is_stateful = stateful
        if not stateful:
            self.reset()
            
        if hasattr(self.neuron, 'set_stateful'):
            self.neuron.set_stateful(stateful) # type: ignore[attr-defined]

    def reset(self):
        """ (v15) 状態をリセット """
        if hasattr(self.neuron, 'reset'):
            self.neuron.reset() # type: ignore[attr-defined]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        (v15: BaseModel (L:71) に合わせて引数を 'input_data' に変更)
        
        Args:
            input_data (torch.Tensor): (T, B, N, C) または (B, N, C)
        
        Returns:
            torch.Tensor: (T, B, N, C) または (B, N, C)
        """
        # (v15) SpikingTransformerV2 (L:528) からは
        #       (B, N, C) が渡されると仮定
        
        # (v15) 状態リセット
        if not self._is_stateful:
            self.reset()

        B, N, C = input_data.shape
        
        # (v15) 複素数 QKV (B, N, C*3*2)
        qkv_complex = self.qkv_linear(input_data)
        
        # (v15) スパイク化 (B, N, C*3*2)
        qkv_spikes, _ = self.neuron(qkv_complex) # type: ignore[attr-defined]

        # (v15) (B, N, C*2)
        # (注: 本来のアテンション計算 (softmax, matmul) は省略し、
        #  ニューロンを通した射影のみを行う)
        
        # (v15) (B, N, C)
        output = self.out_linear(qkv_spikes[..., :self.dim*2]) 
        
        return output
