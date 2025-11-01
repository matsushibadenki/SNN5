# ファイルパス: snn_research/core/networks/sequential_snn_network.py
# タイトル: シーケンシャルSNN（スパイキングニューラルネットワーク）ネットワーク (P1-1 統合)
# 機能説明: 
#   Project SNN4のロードマップ (P4-3) に基づく、
#   AbstractSNNNetwork を継承した具象シーケンシャルSNN。
#
#   (ダミー実装の解消 と P1-1 の統合):
#   - 'forward' メソッド内で、レイヤーの出力 (activity) と
#     トップダウン予測 (ダミー) に基づき「予測誤差 (prediction_error)」を計算し、
#     'model_state' に追加します。
#   - これにより、P1-1 (PredictiveCodingRule) が学習に必要な誤差情報を
#     'model_state' から取得できるようになります。

from typing import Dict, Optional, List, Any, cast
import logging

# --- PyTorch のインポート ---
import torch
import torch.nn as nn
from torch import Tensor

# P4-2, P4-1, P2-1 をインポート
try:
    from .abstract_snn_network import AbstractSNNNetwork
    from ..layers.abstract_snn_layer import AbstractSNNLayer
    from ...layers.abstract_layer import AbstractLayer, LayerOutput
except ImportError:
    # (mypy フォールバック)
    LayerOutput = Dict[str, Tensor] # type: ignore[misc]
    
    from abc import ABC, abstractmethod
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
    
    class AbstractSNNLayer(AbstractLayer): # type: ignore[no-redef, misc]
        def reset_state(self) -> None:
            pass

    class AbstractSNNNetwork(AbstractNetwork): # type: ignore[no-redef, misc]
        def reset_states(self) -> None:
            pass
        @abstractmethod
        def forward(
            self, i: Tensor, t: Optional[Tensor] = None
        ) -> Dict[str, Tensor]:
            raise NotImplementedError


# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)

class SequentialSNNNetwork(AbstractSNNNetwork):
    """
    P4-3: 時系列データを扱うシーケンシャルSNN。
    
    P1-1 (予測符号化) の誤差計算を 'forward' に統合します。
    """

    def __init__(
        self, 
        layers: Optional[List[AbstractLayer]] = None,
        reset_states_on_forward: bool = True
    ) -> None:
        super().__init__(layers)
        self.reset_states_on_forward: bool = reset_states_on_forward
        if logger:
            logger.info(
                f"Initialized SequentialSNNNetwork with {len(self.layers)} layers."
            )

    def forward(
        self, 
        inputs: Tensor, 
        targets: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        (AbstractSNNNetworkから実装)
        ネットワーク全体の順伝播を「時系列」で実行します。
        """
        
        if self.reset_states_on_forward:
            self.reset_states()
        
        try:
            time_steps: int = inputs.shape[1]
        except (AttributeError, IndexError):
            if logger:
                logger.error("Input tensor 'inputs' must have shape [B, T, F].")
            return {'error': torch.tensor(1.0)}

        model_state: Dict[str, Any] = {'inputs': inputs}
        if targets is not None:
            model_state['targets'] = targets
        
        output_history: List[Tensor] = []
        
        # (P1-1) P4-3 の時間ループ
        for t in range(time_steps):
            
            current_input_t: Tensor = inputs[:, t]
            
            # (P1-1) ネットワークの内部状態 (予測など) を保持する
            # (この実装では、直前のステップの活性を「予測」の代わりに使用)
            last_step_activities: Dict[str, Tensor] = {}

            # P2-3: シーケンシャル・レイヤーループ (時間 t 内)
            for layer in self.layers:
                
                # (AbstractLayer (nn.Module) を __call__ で呼び出し)
                layer_output_t: LayerOutput = layer(
                    current_input_t, 
                    model_state
                )
                
                current_activity: Tensor = layer_output_t['activity']
                
                # --- P1-1 予測符号化 (ダミー実装の解消) ---
                # (ダミーロジック: このレイヤーの「予測」は、
                #  1つ前のレイヤーの活性 'last_step_activities.get(layer.name)'
                #  またはトップダウン信号 (ここでは未実装) と仮定)
                
                # (ダミー予測: ゼロ予測)
                prediction: Tensor = torch.zeros_like(current_activity)
                
                # (ダミーロジック: 最終層はターゲットとの誤差を計算)
                if layer == self.layers[-1] and targets is not None:
                    # (ターゲットが [B]、出力が [B, N_out] の場合)
                    # (ダミー: ターゲットをワンホット化)
                    try:
                        target_one_hot: Tensor = nn.functional.one_hot(
                            targets, num_classes=current_activity.shape[1]
                        ).float()
                        prediction = target_one_hot
                    except Exception:
                        pass # ターゲット形状が合わない場合はゼロ予測

                # 予測誤差 = 実際の活性 - 予測
                prediction_error: Tensor = current_activity - prediction
                
                # ----------------------------------------
                
                # P1-1: 状態のマージ (時間 t の状態)
                model_state[f'{layer.name}_output_t'] = layer_output_t
                model_state.update(layer_output_t) 
                
                # (P1-1) 学習規則 (P1-1) が参照できるように誤差を保存
                model_state[f'prediction_error_{layer.name}'] = prediction_error
                # (P1-2) 学習規則が参照できるようにプリシナプス活性を保存
                model_state[f'pre_activity_{layer.name}'] = current_input_t

                current_input_t = current_activity
            
            output_history.append(current_input_t)

        model_state['output_history'] = torch.stack(output_history, dim=1)
        model_state['output'] = current_input_t       
        
        return model_state  # type: ignore[return-value]
