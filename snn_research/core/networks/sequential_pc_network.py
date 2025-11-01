# ファイルパス: snn_research/core/networks/sequential_pc_network.py
# タイトル: シーケンシャル予測符号化ネットワーク (PyTorch準拠)
# 機能説明: 
#   Project SNN4のロードマップ (P2-3) に基づく、
#   AbstractNetwork を継承したシーケンシャルネットワーク。
#
#   (ダミー実装の解消):
#   - Generic[Tensor] および TypeVar('Tensor') を削除。
#   - PyTorch の 'torch.Tensor' を使用します。

from typing import Dict, Optional, List, Any
import logging

# --- ダミー実装の解消 (PyTorch のインポート) ---
from torch import Tensor
# 修正 (エラー 2): nn をインポート
import torch.nn as nn

# P2-2 (抽象ネットワーク) と P2-1 (抽象レイヤー) をインポート
try:
    from ..network import AbstractNetwork
    from ...layers.abstract_layer import AbstractLayer, LayerOutput
except ImportError:
    # (mypy フォールバック)
    LayerOutput = Dict[str, Tensor] # type: ignore[misc]

    from abc import ABC, abstractmethod
    class AbstractLayer(nn.Module, ABC): # type: ignore[no-redef, misc]
        name: str = "dummy"
        def forward(self, i: Tensor, s: Dict[str, Tensor]) -> LayerOutput:
            return {'activity': i} 

    class AbstractNetwork(ABC): # type: ignore[no-redef, misc]
        def __init__(
            self, layers: Optional[List[AbstractLayer]] = None
        ) -> None:
            self.layers: List[AbstractLayer] = \
                layers if layers is not None else []


# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)

class SequentialPCNetwork(AbstractNetwork):
    """
    P2-3: シーケンシャルなレイヤー実行を行うネットワーク。
    """
    
    def __init__(self, layers: Optional[List[AbstractLayer]] = None) -> None:
        super().__init__(layers)
        if logger:
            logger.info(f"Initialized SequentialPCNetwork with {len(self.layers)} layers.")

    def forward(
        self, 
        inputs: Tensor, 
        targets: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        
        model_state: Dict[str, Any] = {'inputs': inputs}
        
        if targets is not None:
            model_state['targets'] = targets

        current_input: Tensor = inputs

        for layer in self.layers:
            try:
                # (AbstractLayer (nn.Module) を __call__ で呼び出し)
                layer_output: LayerOutput = layer(current_input, model_state)
                
                model_state[f'{layer.name}_output'] = layer_output
                model_state.update(layer_output) 
                
                if 'activity' not in layer_output:
                    if logger:
                        logger.warning(
                            f"Layer {layer.name} output missing 'activity'."
                        )
                    current_input = layer_output.get('output', current_input)
                else:
                    current_input = layer_output['activity']

            except Exception as e:
                if logger:
                    logger.error(f"Error during forward pass in layer {layer.name}: {e}")
                return model_state # type: ignore[return-value]

        model_state['output'] = current_input
        return model_state  # type: ignore[return-value]
