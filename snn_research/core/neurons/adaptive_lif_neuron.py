# ファイルパス: snn_research/core/neurons/adaptive_lif_neuron.py
# タイトル: Adaptive Leaky Integrate-and-Fire (LIF) Neuron Module
# 機能説明: 
#   入力電流を受け取り、LIFダイナミクスに基づいて膜電位を更新し、
#   スパイクを生成する nn.Module。
#   (Adaptive という名前だが、現在は動的閾値はオフで運用されている)
#
#   【デバッグ修正】: 
#   - 外部から指定された初期膜電位 (v_init) が反映されず、
#     常にゼロで初期化されるバグを修正。
#   - bias_init を受け取り、入力電流に加算するバイアスとして機能させる。

import logging
from typing import Dict, Any, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor 

# AbstractSNNLayer や BaseModel をインポート (存在する場合)
# (存在しない、あるいは異なるベースクラスの場合は nn.Module のみ)
try:
    # lif_layer.py と同様の抽象クラスを想定
    from ..layers.abstract_snn_layer import AbstractSNNLayer
    BaseNeuronModule = AbstractSNNLayer # type: ignore[misc]
except ImportError:
    # フォールバック
    class BaseNeuronModule(nn.Module): # type: ignore[no-redef, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.time_steps: int = kwargs.get('time_steps', 0)
            self.name: str = kwargs.get('name', "BaseNeuronModule")
            self.built: bool = False
        
        def build(self) -> None:
            self.built = True

        def reset_state(self) -> None:
            pass
        
        def get_total_spikes(self) -> Tensor:
            return torch.tensor(0.0)
        
        def get_total_neurons(self) -> int:
            return 0


# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)


class AdaptiveLIFNeuron(BaseNeuronModule):
    """
    Adaptive LIF Neuron Module (v_init 修正済み)
    
    入力 (電流) を受け取り、膜電位を更新し、スパイク (活動) を返します。
    重み (W) は持たず、状態 (V) のみを管理します。
    """

    def __init__(
        self, 
        features: int,
        threshold: float = 1.0,
        decay: float = 0.95, 
        # デバッグ修正: bias_init を受け取る
        bias_init: float = 0.0,
        # デバッグ修正: 初期膜電位を受け取る
        v_init: float = 0.0,
        name: str = "AdaptiveLIFNeuron",
        **kwargs: Any # 他の未使用パラメータ (threshold_decay など) を吸収
    ) -> None:
        
        super().__init__(name=name, **kwargs)
        
        self.decay: float = decay
        self.threshold: float = threshold
        # デバッグ修正: v_init をインスタンス変数として保存
        self.v_init: float = v_init
        
        self._features: int = features
        
        # デバッグ修正: bias_init の値をバイアスパラメータとして設定
        # (LIFLayer と異なり、requires_grad=True の可能性があるが、
        # HPOのデバッグ設定 (2.0) を反映するため False で固定)
        self.b: nn.Parameter = nn.Parameter(
            torch.full((self._features,), bias_init),
            requires_grad=False
        )
        
        self.membrane_potential: Optional[Tensor] = None
        self.total_spikes: Tensor = torch.tensor(0.0)
        
        self.build()

    def build(self) -> None:
        """
        パラメータを初期化します。(このクラスでは主にバイアス)
        """
        # (重みWの初期化は不要)
        # (学習規則のセットアップは不要)
        
        self.built = True

    def _init_state(self, batch_size: int, device: torch.device) -> None:
        if logger:
            logger.debug(f"Initializing state for {self.name} with batch size {batch_size}")
        
        # デバッグ修正: torch.zeros を使用せず、self.v_init で指定された値で初期化する
        # (B, N, C) または (B, F) の形状に対応
        shape = (batch_size, self._features)
        
        # (B, N, C) の場合 (Transformer 用)
        # self.membrane_potential が None でない場合、その形状を使う
        if self.membrane_potential is not None:
             # (Pdb)
             # V_t_minus_1 が (B, N, C) の場合、(B, N, C) で初期化
             if self.membrane_potential.dim() > 2:
                 # (B, N, C)
                 shape = self.membrane_potential.shape # type: ignore[assignment]
             else:
                 # (B, C)
                 shape = (batch_size, self._features) # type: ignore[assignment]

        self.membrane_potential = torch.full(
            shape, # type: ignore[arg-type]
            fill_value=self.v_init, 
            device=device
        )

    def forward(
        self, 
        inputs: Tensor # 入力電流 I_t (B, N, C) または (B, F)
    ) -> Tensor: # 出力スパイク (B, N, C) または (B, F)
        
        if not self.built:
            raise RuntimeError(f"Layer {self.name} has not been built.")
        
        # (B, N, C) または (B, F)
        batch_size = inputs.shape[0]
        
        # (B, N, C) の場合、_init_state が (B, C) で初期化しないように形状を渡す
        if self.membrane_potential is None or self.membrane_potential.shape[0] != batch_size:
            # (Pdb)
            # (B, N, C) の形状を正しく渡すために
            if inputs.dim() > 2:
                self.membrane_potential = torch.full(
                    inputs.shape, # (B, N, C)
                    fill_value=self.v_init, 
                    device=inputs.device
                )
            else:
                self._init_state(batch_size, inputs.device)
        
        V_t_minus_1: Tensor = cast(Tensor, self.membrane_potential)

        # --- LIFダイナミクス (ゼロへのハードリセット) ---
        
        # 1. 入力電流にバイアスを加算
        # (B, N, C) + (C,) -> (B, N, C) (ブロードキャスト)
        I_t_biased: Tensor = inputs + self.b
        
        # 2. 膜電位のリーク
        V_leaked: Tensor = V_t_minus_1 * self.decay
        
        # 3. 膜電位の更新
        V_new: Tensor = V_leaked + I_t_biased
        
        # 4. スパイクの生成
        spikes_t: Tensor = (V_new > self.threshold).float()
        
        # 5. 膜電位のリセット (ゼロへのハードリセット)
        V_reset: Tensor = V_new * (1.0 - spikes_t)
        
        # 状態の更新
        self.membrane_potential = V_reset
        
        # スパイク数をカウント (BaseModel のため)
        self.total_spikes += torch.sum(spikes_t)
        
        # スパイク (活動) を返す
        return spikes_t

    def reset_state(self) -> None:
        if logger:
            logger.debug(f"Resetting state for {self.name}")
        self.membrane_potential = None
        self.total_spikes = torch.tensor(0.0)

    # BaseModel 互換のためのメソッド
    def get_total_spikes(self) -> Tensor:
        return self.total_spikes

    def get_total_neurons(self) -> int:
        return self._features
