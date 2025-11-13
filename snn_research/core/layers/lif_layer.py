# ファイルパス: snn_research/core/layers/lif_layer.py
# タイトル: Leaky Integrate-and-Fire (LIF) SNNレイヤー

import logging
from typing import Dict, Any, Optional, Tuple, cast

# --- PyTorch のインポート ---
import torch
import torch.nn as nn
from torch import Tensor 

# ... (中略：インポートとダミークラスの定義は省略) ...

@torch.jit.script
def lif_update(
    inputs: Tensor, 
    V: Tensor, 
    W: Tensor, 
    b: Tensor, 
    decay: float, 
    threshold: float
) -> Tuple[Tensor, Tensor]:
    """ P4-4: 単一ステップのLIFダイナミクス (PyTorch 実装) """
    I_t: Tensor = nn.functional.linear(inputs, W, b)
    V_leaked: Tensor = V * decay
    V_new: Tensor = V_leaked + I_t
    spikes: Tensor = (V_new > threshold).float()
    
    # 【ロジック修正: ゼロへのハードリセット】
    # 発火したニューロンの電位を、閾値減算ではなく、問答無用で 0 にリセットします。
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
        # 【バグ修正: バイアス初期値の追加】
        bias_init: float = 0.0, 
    ) -> None:
        
        dummy_shape: Tuple[int, ...] = (0,)
        super().__init__(dummy_shape, dummy_shape, learning_config, name)
        
        self.decay: float = decay
        self.threshold: float = threshold
        
        self._input_features: int = input_features
        self._neurons: int = neurons
        
        self.W: nn.Parameter = nn.Parameter(
            torch.empty(self._neurons, self._input_features), 
            requires_grad=False
        )
        
        # 【バグ修正: self.b を bias_init で初期化】
        # nn.Parameter の初期化時に、指定されたバイアス値を使用します。
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
        # nn.init.zeros_(self.b) # 【バグ修正: ゼロ初期化を削除】
                                # self.b は __init__ で既に初期化済み
        
        # P1-4: 学習可能なパラメータとして登録
        self.params = [self.W, self.b]
        
        # ... (中略：学習規則の設定は変更なし) ...
            
        self.built = True
        
    # ... (以降のメソッドは変更なし) ...
