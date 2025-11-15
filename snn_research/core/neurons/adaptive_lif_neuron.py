# ファイルパス: snn_research/core/neurons/adaptive_lif_neuron.py
# タイトル: Adaptive Leaky Integrate-and-Fire (LIF) Neuron Module
# 機能説明: 
#   入力電流を受け取り、LIFダイナミクスに基づいて膜電位を更新し、
#   スパイクを生成する nn.Module。

import logging
from typing import Dict, Any, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor 

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
            pass

logger: logging.Logger = logging.getLogger(__name__)

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
        bias_init: float = 0.0,
        v_init: float = 0.0,
        time_steps: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(time_steps=time_steps, **kwargs) 
        
        self.features = int(features)
        self.threshold = threshold
        self.decay = decay
        
        # 学習可能なバイアス項
        self.b = nn.Parameter(
            torch.full((features,), bias_init, dtype=torch.float)
        )
        
        # 初期膜電位
        self.v_init = v_init
        
        # 膜電位（状態）
        self.membrane_potential: Optional[Tensor] = None
        
        # BaseModel用のニューロン数
        self._total_neurons = self.features
        
        self.built = True

    def reset_state(self) -> None:
        """
        推論時またはエポック開始時に状態をリセット
        """
        self.membrane_potential = None

    def forward(self, inputs: Tensor) -> Tensor:
        """
        LIFダイナミクスを適用
        
        Args:
            inputs: (B, ..., C) 形状の入力電流。最後の次元は self.features と一致
        
        Returns:
            (B, ..., C) 形状の出力スパイク
        """
        if not self.built:
            raise RuntimeError(f"Layer {self.name} has not been built.")
        
        # 入力形状の検証
        if inputs.shape[-1] != self.features:
            raise ValueError(
                f"Input last dimension {inputs.shape[-1]} does not match "
                f"features {self.features}"
            )
        
        # 1. 入力電流にバイアスを加算
        # (B, ..., C) + (C,) -> (B, ..., C)
        I_t_biased: Tensor = inputs + self.b
        
        # 2. 前ステップの膜電位を処理
        if self.membrane_potential is None:
            # t=0: v_init を decay なしで使用
            V_t_minus_1_decayed = torch.full_like(
                inputs,
                fill_value=self.v_init
            )
        else:
            # t>0: 既存の膜電位を decay
            V_t_minus_1: Tensor = cast(Tensor, self.membrane_potential)
            V_t_minus_1_decayed = V_t_minus_1 * self.decay
        
        # 3. 膜電位の更新（積分）
        V_new: Tensor = V_t_minus_1_decayed + I_t_biased
        
        # 4. スパイクの生成
        spikes_t: Tensor = (V_new > self.threshold).float()
        
        # 5. 膜電位のリセット（スパイク位置をゼロに）
        V_reset: Tensor = V_new * (1.0 - spikes_t)
        
        # 状態の更新
        self.membrane_potential = V_reset
        
        # スパイクカウント（BaseModel用）
        if hasattr(self, 'total_spikes'):
            self.total_spikes = self.total_spikes + torch.sum(spikes_t)
        
        return spikes_t
