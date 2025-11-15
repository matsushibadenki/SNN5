# ファイルパス: snn_research/core/neurons/adaptive_lif_neuron.py
# タイトル: Adaptive Leaky Integrate-and-Fire (LIF) Neuron Module
# 機能説明: 
#   入力電流を受け取り、LIFダイナミクスに基づいて膜電位を更新し、
#   スパイクを生成する nn.Module。
#
#   【修正点】:
#   - (中略)
#
#   【!!! スパイク消滅 (spike_rate=0) 修正 v3 !!!】
#   - L.101-142 の forward ロジックを修正。
#   - t=0 (self.membrane_potential is None) の時、v_init (0.4995) が
#     入力 (inputs) より先にディケイ (decay) されてしまい、
#     V_new が 0.4845 + inputs となり閾値 0.5 を超えられない問題を修正。
#   - t=0 の時はディケイを適用せず、V_new = v_init + inputs + bias となるように
#     ロジックを変更する。

import logging
from typing import Dict, Any, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor 

# SpikingTransformerV2 (v2.py) で使われている BaseModel をインポート
try:
    from snn_research.core.base import BaseModel
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("BaseModel not found. Falling back to simple nn.Module.")
    class BaseModel(nn.Module): # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.time_steps: int = kwargs.get('time_steps', 0)
            self._total_neurons: int = 0
            self.total_spikes: Tensor = torch.tensor(0.0)

        def get_total_neurons(self) -> int:
            return self._total_neurons

        def get_total_spikes(self) -> Tensor:
            return self.total_spikes

        def reset_state(self) -> None:
            pass # サブクラスで実装

logger: logging.Logger = logging.getLogger(__name__)

# BaseModelを継承することで、get_total_spikes() などが機能する
class AdaptiveLIFNeuron(BaseModel): 
    """
    LIFニューロンのダイナミクスをカプセル化する nn.Module。
    SpikingTransformerV2 などのアーキテクチャ内で「活性化関数」のように振る舞う。
    """
    def __init__(
        self, 
        features: int, 
        threshold: float = 1.0, 
        decay: float = 0.95,
        # --- ▼ 修正 (ステップ 2 & 4) ▼ ---
        bias_init: float = 0.0,
        v_init: float = 0.0,
        # --- ▲ 修正 ▲ ---
        time_steps: int = 0, # BaseModel のために追加
        **kwargs: Any
    ) -> None:
        # time_steps を BaseModel の __init__ に渡す
        super().__init__(time_steps=time_steps, **kwargs) 
        
        self.features = int(features) # HPO対策でint()キャスト
        self.threshold = threshold
        self.decay = decay
        
        # --- ▼ 修正 (ステップ 2) ▼ ---
        # 外部から指定されたバイアスで初期化
        # (注意: このバイアスは重み W とは別です)
        self.b = nn.Parameter(
            torch.full((features,), bias_init, dtype=torch.float)
        )
        # --- ▲ 修正 ▲ ---

        # --- ▼ 修正 (ステップ 4) ▼ ---
        # 外部から指定された初期膜電位
        self.v_init = v_init
        # --- ▲ 修正 ▲ ---
        
        # 膜電位 (状態)
        self.membrane_potential: Optional[Tensor] = None
        
        # BaseModel の get_total_neurons() のためにニューロン数を登録
        self._total_neurons = self.features
        
        self.built = True

    def _init_state(self, batch_size: int, device: torch.device) -> None:
        """
        （B, C）の形状で膜電位を初期化します。
        (注: このメソッドは v3 修正により forward 内で直接処理されるため、
         現在は使用されていません。)
        """
        shape = (batch_size, self.features)
        
        # --- ▼ 修正 (ステップ 4) ▼ ---
        # 常にゼロで初期化するのではなく、self.v_init を使用
        self.membrane_potential = torch.full(
            shape, 
            fill_value=self.v_init, 
            device=device,
            dtype=torch.float
        )
        # --- ▲ 修正 ▲ ---

    def reset_state(self) -> None:
        """
        推論時またはエポック開始時に状態をリセットします。
        """
        self.membrane_potential = None

    def forward(self, inputs: Tensor) -> Tensor:
        """
        LIFダイナミクスを適用します。
        
        Args:
            inputs (Tensor): (B, ..., C) の形状を持つ入力電流 (I_t)。
                             C は self.features と一致する必要があります。
        
        Returns:
            Tensor: (B, ..., C) の形状を持つ出力スパイク (S_t)。
        """
        if not self.built:
            raise RuntimeError(f"Layer {self.name} has not been built.")
        
        # --- ▼▼▼ 【!!! spike_rate=0 修正 v3 !!!】 ▼▼▼
        
        # 1. 入力電流にバイアスを加算
        # (B, ..., C) + (C,) -> (B, ..., C) (ブロードキャスト)
        I_t_biased: Tensor = inputs + self.b
        
        V_t_minus_1_decayed: Tensor

        if self.membrane_potential is None:
            # t=0 (状態が未初期化) の場合
            # v_init (0.4995) をディケイ(decay)させずにそのまま使用する
            
            # (B, N, C) のようなViTの形状に対応
            if inputs.dim() > 2:
                V_t_minus_1_decayed = torch.full(
                    inputs.shape, # (B, N, C)
                    fill_value=self.v_init, 
                    device=inputs.device,
                    dtype=torch.float
                )
            else: # (B, C)
                 V_t_minus_1_decayed = torch.full(
                    (inputs.shape[0], self.features), 
                    fill_value=self.v_init, 
                    device=inputs.device,
                    dtype=torch.float
                )
        else:
            # t > 0 の場合
            # 既存の膜電位 (V_t_minus_1) をディケイさせる
            V_t_minus_1: Tensor = cast(Tensor, self.membrane_potential)
            V_t_minus_1_decayed = V_t_minus_1 * self.decay

        # 2. 膜電位の更新 (積分)
        # (t=0): V_new = v_init + (inputs + bias)
        # (t>0): V_new = (V_t-1 * decay) + (inputs + bias)
        V_new: Tensor = V_t_minus_1_decayed + I_t_biased
        
        # 3. スパイクの生成
        spikes_t: Tensor = (V_new > self.threshold).float()
        
        # 4. 膜電位のリセット (ゼロへのハードリセット)
        V_reset: Tensor = V_new * (1.0 - spikes_t)
        
        # --- ▲▲▲ 【!!! spike_rate=0 修正 v3 !!!】 ▲▲▲
        
        # 次のタイムステップのために状態を更新
        self.membrane_potential = V_reset
        
        # BaseModel のスパイクカウント機構 (get_total_spikes) のため
        if hasattr(self, 'total_spikes'):
            self.total_spikes += torch.sum(spikes_t)
        
        return spikes_t
