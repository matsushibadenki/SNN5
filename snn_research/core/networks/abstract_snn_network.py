# ファイルパス: snn_research/core/networks/abstract_snn_network.py
# タイトル: 抽象SNN（スパイキングニューラルネットワーク）ネットワーク (PyTorch準拠)
# 機能説明: 
#   Project SNN4のロードマップ (P4-2) に基づく、
#   SNN特有の時系列処理と状態リセットを管理する抽象ネットワーク。
#
#   (ダミー実装の解消):
#   - Generic[Tensor] および TypeVar('Tensor') を削除。
#   - PyTorch の 'torch.Tensor' を使用します。

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
import logging

# --- ダミー実装の解消 (PyTorch のインポート) ---
import torch.nn as nn
from torch import Tensor

# P2-2 (抽象ネットワーク) と P4-1 (抽象SNNレイヤー) をインポート
try:
    from ..network import AbstractNetwork
    from ..layers.abstract_snn_layer import AbstractSNNLayer
    from ...layers.abstract_layer import AbstractLayer, LayerOutput
except ImportError:
    # (mypy フォールバック)
    LayerOutput = Dict[str, Tensor] # type: ignore[misc]
    
    # 修正 (エラー 23-25): [Tensor] と Generic を削除
    class AbstractLayer(nn.Module, ABC): # type: ignore[no-redef, misc]
        name: str = "dummy"
        def forward(self, i: Tensor, s: Dict[str, Tensor]) -> LayerOutput:
            return {'activity': i}
        def build(self) -> None: pass

    class AbstractNetwork(ABC): # type: ignore[no-redef, misc]
        def __init__(
            self, layers: Optional[List[AbstractLayer]] = None
        ) -> None:
            self.layers: List[AbstractLayer] = \
                layers if layers is not None else []
        @abstractmethod
        def forward(
            self, i: Tensor, t: Optional[Tensor] = None
        ) -> Dict[str, Tensor]:
            raise NotImplementedError
    
    # 修正 (エラー 26): [Tensor] を削除
    class AbstractSNNLayer(AbstractLayer): # type: ignore[no-redef, misc]
        def reset_state(self) -> None:
            pass


# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)

# 修正 (エラー 27): [Tensor] を削除
class AbstractSNNNetwork(AbstractNetwork):
    """
    P4-2: SNNのための時系列処理ネットワーク (抽象クラス)。
    """
    
    # 修正 (エラー 28): [Tensor] を削除
    def __init__(self, layers: Optional[List[AbstractLayer]] = None) -> None:
        super().__init__(layers)

    def reset_states(self) -> None:
        """
        P4-2: ネットワーク内の全てのSNNレイヤーの状態をリセットします。
        """
        if logger:
            logger.debug("Resetting SNN layer states...")
        for layer in self.layers:
            if isinstance(layer, AbstractSNNLayer):
                layer.reset_state()

    @abstractmethod
    def forward(
        self, 
        inputs: Tensor, 
        targets: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        (AbstractNetworkからオーバーライド)
        ネットワーク全体の順伝播を「時系列」で実行します。
        """
        raise NotImplementedError
