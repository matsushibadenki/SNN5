# ファイルパス: snn_research/core/layers/lif_layer.py
# タイトル: Leaky Integrate-and-Fire (LIF) SNNレイヤー
# 機能説明: 
#   AbstractSNNLayer (P4-1) を継承する具象LIFレイヤー。
#
#   【緊急修正 P4-4 / バグ修正 P1-1 連携】:
#   1. スパイクが発生しない問題を解決するため、LIFのダイナミクスを
#      「閾値減算リセット」から、より安定した「ゼロへの**ハードリセット**」に変更。
#   2. モデル設定で強制されたバイアス値 (例: NEURON_BIAS) が
#      build()内でゼロに上書きされるバグを修正するため、
#      __init__でバイアス初期値を受け取り、build()でのゼロ初期化を削除。

import logging
from typing import Dict, Any, Optional, Tuple, cast

# --- PyTorch のインポート ---
import torch
import torch.nn as nn
from torch import Tensor 

# P4-1 (抽象SNNレイヤー) と P1 のモジュールをインポート
try:
    from .abstract_snn_layer import AbstractSNNLayer
    from ...layers.abstract_layer import LayerOutput
    from ...config.learning_config import BaseLearningConfig
    from ..learning_rule import Parameters
    from ..learning_rules.predictive_coding_rule import PredictiveCodingRule
except ImportError:
    # (mypy フォールバック - P1-3, P4-1 関連のダミー定義)
    LayerOutput = Dict[str, Tensor] # type: ignore[misc]
    BaseLearningConfig = Any # type: ignore[misc, assignment]
    Parameters = Any # type: ignore[misc, assignment]
    from abc import ABC, abstractmethod
    class AbstractLayer(nn.Module, ABC): # type: ignore[no-redef, misc]
         def __init__(self, i: Any, o: Any, l: Any, n: str) -> None: super().__init__()
    class AbstractSNNLayer(AbstractLayer): # type: ignore[no-redef, misc]
        def __init__(self, i: Any, o: Any, l: Any, n: str) -> None: 
            super().__init__(i, o, l, n)
        def build(self) -> None: raise NotImplementedError
        def forward(self, i: Tensor, s: Dict[str, Tensor]) -> LayerOutput:
            raise NotImplementedError
        def reset_state(self) -> None: pass
        params: Parameters = []
        learning_config: Optional[BaseLearningConfig] = None
        learning_rule: Any = None
    PredictiveCodingRule = Any # type: ignore[misc, assignment]

# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)

@torch.jit.script
def lif_update(
    inputs: Tensor, 
    V: Tensor, 
    W: Tensor, 
    b: Tensor, 
    decay: float, 
    threshold: float
) -> Tuple[Tensor, Tensor]:
    """ 
    P4-4: 単一ステップのLIFダイナミクス (PyTorch 実装) 
    (修正済み: ゼロへのハードリセットを採用)
    """
    I_t: Tensor = nn.functional.linear(inputs, W, b)
    V_leaked: Tensor = V * decay
    V_new: Tensor = V_leaked + I_t
    spikes: Tensor = (V_new > threshold).float()
    
    # 【ロジック修正 P4-4 (スパイク安定化)】: 
    # 閾値減算リセット (V_new - (spikes * threshold)) から
    # ゼロへのハードリセット (V_new * (1.0 - spikes)) に変更。
    # 発火したニューロンの電位を次のステップで確実に 0 にリセットします。
    V_reset: Tensor = V_new * (1.0 - spikes)
    
    return V_reset, spikes


class LIFLayer(AbstractSNNLayer):
    """
    P4-4: Leaky Integrate-and-Fire (LIF) レイヤー (PyTorch実装)。
    """

    def __init__(
        self, 
        input_features: int, 
        neurons: int,
        learning_config: Optional[BaseLearningConfig] = None,
        name: str = "LIFLayer",
        decay: float = 0.95, 
        threshold: float = 1.0,
        # 【バグ修正 P1-1 連携】: バイアスの初期値を受け取る
        bias_init: float = 0.0,
    ) -> None:
        
        dummy_shape: Tuple[int, ...] = (0,)
        # (name は AbstractLayer の __init__ で設定される)
        super().__init__(dummy_shape, dummy_shape, learning_config, name)
        
        self.decay: float = decay
        self.threshold: float = threshold
        
        self._input_features: int = input_features
        self._neurons: int = neurons
        
        self.W: nn.Parameter = nn.Parameter(
            torch.empty(self._neurons, self._input_features), 
            requires_grad=False
        )
        # 【バグ修正 P1-1 連携】: bias_init の値を初期値として設定
        self.b: nn.Parameter = nn.Parameter(
            torch.full((self._neurons,), bias_init),
            requires_grad=False
        )
        
        self.membrane_potential: Optional[Tensor] = None


    def build(self) -> None:
        """
        (P2-1) パラメータを初期化し、(P1-4) 学習規則をセットアップします。
        """
        if logger:
            logger.debug(f"Building layer: {self.name}")
            
        nn.init.kaiming_uniform_(self.W, a=0.01)
        # 【バグ修正 P1-1 連携】: __init__ で設定されたカスタムバイアスを
        # 上書きしないよう、このゼロ初期化を削除します。
        # nn.init.zeros_(self.b) 
        
        # P1-4: 学習可能なパラメータとして登録
        self.params = [self.W, self.b]
        
        # P1-4: 学習規則のインスタンス化
        if self.learning_config:
            # (P1-1 の PC ルールをデフォルトで使用)
            rule_cls: Any = PredictiveCodingRule
            
            # --- ダミー実装の解消 (P1-1 / P4-4 連携) ---
            # P1-3 の設定を取得
            rule_kwargs: Dict[str, Any] = self.learning_config.to_dict()
            # P1-4 (AbstractLearningRule) のため、レイヤー名を渡す
            rule_kwargs['layer_name'] = self.name
            
            self.learning_rule = rule_cls(
                self.params, 
                **rule_kwargs
            )
            
        self.built = True

    def _init_state(self, batch_size: int, device: torch.device) -> None:
        if logger:
            logger.debug(f"Initializing state for {self.name} with batch size {batch_size}")
        
        self.membrane_potential = torch.zeros(
            (batch_size, self._neurons), device=device
        )

    def forward(
        self, 
        inputs: Tensor, 
        model_state: Dict[str, Tensor]
    ) -> LayerOutput:
        if not self.built:
            raise RuntimeError(f"Layer {self.name} has not been built.")
        
        batch_size: int = inputs.shape[0]

        if self.membrane_potential is None:
            self._init_state(batch_size, inputs.device)
        
        V_t_minus_1: Tensor = cast(Tensor, self.membrane_potential)

        V_t: Tensor
        spikes_t: Tensor
        V_t, spikes_t = lif_update(
            inputs, V_t_minus_1, self.W, self.b, self.decay, self.threshold
        )
        
        self.membrane_potential = V_t
        
        return {
            'activity': spikes_t, # (スパイク)
            'membrane_potential': V_t
        }

    def reset_state(self) -> None:
        if logger:
            logger.debug(f"Resetting state for {self.name}")
        self.membrane_potential = None
