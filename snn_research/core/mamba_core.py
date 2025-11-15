# ファイルパス: snn_research/core/mamba_core.py
# Title: SNN State Space Model (SSM) Core
#
# 機能の説明: MambaアーキテクチャのコアコンポーネントであるSSMを
# SNN（Spiking Neural Network）で実装したモジュール。
#
# 【修正内容 v29: 循環インポート (Circular Import) の修正】
# - health-check 実行時に 'ImportError: ... (most likely due to a circular import)'
#   が発生する問題に対処します。
# - (L: 19) 'from snn_research.core.snn_core import SNNCore' が、
#   snn_core.py (L:28) -> ... -> mamba_core.py (L:19) という
#   循環参照を引き起こしていました。
# - (L: 22) 'SNNCore' を継承するのは誤りであり、
#   'BaseModel' に修正しました。
# - (L: 19) 'SNNCore' のインポートを削除し、'from .base import BaseModel' を
#   インポートするように変更しました。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .neurons import get_neuron_by_name

# --- ▼▼▼ 【!!! 修正 v29: 循環インポート修正 !!!】 ▼▼▼
# (from snn_research.core.snn_core import SNNCore を削除)
from .base import BaseModel # BaseModel をインポート

class SNN_SSM(BaseModel): # 'SNNCore' -> 'BaseModel' に変更
# --- ▲▲▲ 【!!! 修正 v29】 ▲▲▲
    """
    Spiking Neural Network State Space Model (SNN_SSM)
    (中略)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int,
        time_steps: int,
        neuron_config: Dict[str, Any],
        **kwargs # (v15: BaseModel から vocab_size を吸収)
    ):
        # (v15: BaseModel の __init__ を呼び出す)
        super(SNN_SSM, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.d_state = d_state
        self.time_steps = time_steps
        self.neuron_config = neuron_config

        # SSM パラメータ
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, 1))
        self.C = nn.Parameter(torch.randn(1, d_state))
        
        # ニューロンの初期化
        neuron_config_ssm = neuron_config.copy()
        neuron_config_ssm['features'] = d_state
        self.neuron = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_ssm
        )
        
        neuron_config_output = neuron_config.copy()
        neuron_config_output['features'] = d_model
        self.output_neuron = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_output
        )

        # (v15) 状態管理
        self._is_stateful = False
        self.built = True
        self.state = torch.zeros(1, d_state, device=self.A.device) # (仮: デバイス指定)

    def set_stateful(self, stateful: bool):
        """ (v15) 状態管理モードを設定 """
        self._is_stateful = stateful
        if not stateful:
            self.reset()
            
        # (v15) SpikingTransformerV2 (L:323) に倣い、
        #       ニューロンのリセット/状態設定を伝播
        if hasattr(self.neuron, 'set_stateful'):
            self.neuron.set_stateful(stateful) # type: ignore[attr-defined]
        if hasattr(self.output_neuron, 'set_stateful'):
            self.output_neuron.set_stateful(stateful) # type: ignore[attr-defined]

    def reset(self):
        """ (v15) 状態をリセット """
        self.state = torch.zeros_like(self.state)
        if hasattr(self.neuron, 'reset'):
            self.neuron.reset() # type: ignore[attr-defined]
        if hasattr(self.output_neuron, 'reset'):
            self.output_neuron.reset() # type: ignore[attr-defined]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        (v15: BaseModel (L:71) に合わせて引数を 'input_data' に変更)
        
        Args:
            input_data (torch.Tensor): (T, B, N, C) - C = d_model
        
        Returns:
            torch.Tensor: (T, B, N, C) - 出力スパイク
        """
        T, B, N, C = input_data.shape
        
        # (v15) 状態リセット
        if not self._is_stateful:
            self.reset()
            
        # (v15) デバイスを input_data に合わせる
        if self.state.device != input_data.device:
            self.state = self.state.to(input_data.device)
            
        # (v15) バッチサイズ (B*N) に合わせて状態を拡張
        current_state = self.state.repeat(B * N, 1).view(B, N, C, self.d_state)

        outputs = []
        for t in range(T):
            x_t = input_data[t] # (B, N, C)
            
            # (v15) (B, N, C, 1) に拡張
            x_t_expanded = x_t.unsqueeze(-1) 
            
            # (v15) (B, N, C, D_state)
            state_update = torch.einsum('bni,id->bnid', x_t, self.A) + \
                           torch.einsum('bni,id->bnid', x_t_expanded, self.B)
            
            # (v15) ニューロンを適用 (B, N, C, D_state) -> (B, N, C, D_state)
            current_state, _ = self.neuron(state_update) # type: ignore[attr-defined]
            
            # (v15) 出力計算 (B, N, C, 1)
            output_update = torch.einsum('bnid,id->bni', current_state, self.C)
            
            # (v15) (B, N, C)
            output_update = output_update.squeeze(-1) 
            
            # (v15) 出力ニューロン (B, N, C) -> (B, N, C)
            output_spike, _ = self.output_neuron(output_update) # type: ignore[attr-defined]
            
            outputs.append(output_spike)

        # (v15) 状態の保存
        if self._is_stateful:
            # (B, N, C, D_state) -> (1, D_state) (平均化)
            self.state = torch.mean(current_state, dim=(0, 1, 2)).detach()
            
        return torch.stack(outputs, dim=0)
