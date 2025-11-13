# ファイルパス: snn_research/core/neurons/adaptive_lif_neuron.py
# タイトル: Adaptive Leaky Integrate-and-Fire (LIF) Neuron Module
# 機能説明: 
#   入力電流を受け取り、LIFダイナミクスに基づいて膜電位を更新し、
#   スパイクを生成する nn.Module。
#   (SpikingTransformerV2 が実際に使用するニューロンモジュール)
#
#   【修正点】:
#   - `lif_layer.py` からデバッグ履歴の修正 (ステップ 1, 2, 4) を移植。
#   - ステップ1: ゼロへのハードリセット (V_reset = V_new * (1.0 - spikes_t)) を実装。
#   - ステップ2: `bias_init` を受け取り、`self.b` を初期化。
#   - ステップ4: `v_init` を受け取り、`self.membrane_potential` の初期値として使用。
#   - (注意: ステップ5 (Xavier初期化) は、このファイルには適用できません。
#     このモジュールは重み(W)を持たないためです。)

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
        
        batch_size: int = inputs.shape[0]

        # 状態が未初期化 (t=0) の場合、v_init を使って初期化
        if self.membrane_potential is None:
            # --- ▼ 修正 (ステップ 4) ▼ ---
            # (B, N, C) のようなViTの形状に対応
            if inputs.dim() > 2:
                self.membrane_potential = torch.full(
                    inputs.shape, # (B, N, C)
                    fill_value=self.v_init, 
                    device=inputs.device,
                    dtype=torch.float
                )
            else:
                self._init_state(batch_size, inputs.device) # (B, C)
            # --- ▲ 修正 ▲ ---
        
        V_t_minus_1: Tensor = cast(Tensor, self.membrane_potential)

        # --- ▼ 修正 (ステップ 1: ゼロへのハードリセット) ▼ ---
        
        # 1. 入力電流にバイアスを加算
        # (B, ..., C) + (C,) -> (B, ..., C) (ブロードキャスト)
        I_t_biased: Tensor = inputs + self.b
        
        # 2. 膜電位のリーク
        V_leaked: Tensor = V_t_minus_1 * self.decay
        
        # 3. 膜電位の更新 (積分)
        V_new: Tensor = V_leaked + I_t_biased
        
        # 4. スパイクの生成
        spikes_t: Tensor = (V_new > self.threshold).float()
        
        # 5. 膜電位のリセット (ゼロへのハードリセット)
        V_reset: Tensor = V_new * (1.0 - spikes_t)
        
        # --- ▲ 修正 ▲ ---
        
        # 次のタイムステップのために状態を更新
        self.membrane_potential = V_reset
        
        # BaseModel のスパイクカウント機構 (get_total_spikes) のため
        if hasattr(self, 'total_spikes'):
            self.total_spikes += torch.sum(spikes_t)
        
        return spikes_t
