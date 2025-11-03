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
#
#   (改善 v2):
#   - P1-1 のダミー実装（ゼロ予測）を、実際の「誤差逆伝播（ローカル版）」に置き換え。
#   - 順伝播で活性を計算・保存した後、最終層の誤差を計算し、
#     各層の重みの転置（W.T）を使って誤差をトップダウンに伝播させ、
#     'prediction_error_{layer.name}' として model_state に保存する。

from typing import Dict, Optional, List, Any, cast
import logging

# --- PyTorch のインポート ---
import torch
import torch.nn as nn
from torch import Tensor

# P4-2, P4-1, P2-1 をインポート
try:
    from .abstract_snn_network import AbstractSNNNetwork
    # --- ▼ 修正: LIFLayer (具象クラス) をインポート (W.T を使うため) ▼ ---
    from ..layers.lif_layer import LIFLayer
    from ...layers.abstract_layer import AbstractLayer, LayerOutput
    # --- ▲ 修正 ▲ ---
except ImportError:
    # (mypy フォールバック)
    LayerOutput = Dict[str, Tensor] # type: ignore[misc]
    
    from abc import ABC, abstractmethod
    class AbstractLayer(nn.Module, ABC): # type: ignore[no-redef, misc]
        name: str = "dummy"
        def forward(self, i: Tensor, s: Dict[str, Tensor]) -> LayerOutput:
            return {'activity': i}
        def build(self) -> None: pass
        # --- ▼ 修正: W を追加 ▼ ---
        W: nn.Parameter = nn.Parameter(torch.empty(0)) # type: ignore[assignment]
        # --- ▲ 修正 ▲ ---

    class AbstractNetwork(ABC): # type: ignore[no-redef, misc]
        def __init__(
            self, layers: Optional[List[AbstractLayer]] = None
        ) -> None:
            self.layers: List[AbstractLayer] = \
                layers if layers is not None else []
    
    class AbstractSNNLayer(AbstractLayer): # type: ignore[no-redef, misc]
        def reset_state(self) -> None:
            pass
            
    # --- ▼ 修正: LIFLayer のダミーを定義 ▼ ---
    class LIFLayer(AbstractSNNLayer): # type: ignore[no-redef, misc]
        pass
    # --- ▲ 修正 ▲ ---

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
        
        (改善 v2): 順伝播後に誤差の逆伝播（ローカル版）を実行し、
                   P1-1 のための 'prediction_error' を計算します。
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
                
                # (P1-2) 学習規則が参照できるようにプリシナプス活性を保存
                model_state[f'pre_activity_{layer.name}'] = current_input_t
                
                # (AbstractLayer (nn.Module) を __call__ で呼び出し)
                layer_output_t: LayerOutput = layer(
                    current_input_t, 
                    model_state
                )
                
                current_activity: Tensor = layer_output_t['activity']
                
                # P1-1: 状態のマージ (時間 t の状態)
                model_state[f'{layer.name}_output_t'] = layer_output_t
                model_state.update(layer_output_t) 
                
                current_input_t = current_activity
            
            output_history.append(current_input_t)

        model_state['output_history'] = torch.stack(output_history, dim=1)
        model_state['output'] = current_input_t       
        
        # --- ▼▼▼ P1-1 改善 (ダミー実装の解消) ▼▼▼ ---
        # 最終時間ステップ (T) の状態に基づき、誤差を計算し、逆伝播させる
        
        final_output: Tensor = current_input_t # 最終層の最終時間ステップの活性 (B, N_out)
        
        # (P1-1) 学習規則 (PredictiveCodingRule) のための誤差計算
        # (BPTTのローカル近似として、最終ステップの誤差を逆伝播させる)
        
        # 1. 最終層の誤差を計算
        top_down_error: Optional[Tensor] = None
        
        if targets is not None:
            try:
                # ターゲット (B,) をワンホット化 (B, N_out)
                target_one_hot: Tensor = nn.functional.one_hot(
                    targets, num_classes=final_output.shape[1]
                ).float()
                
                # 最終層の誤差 = 活性 - ターゲット
                top_down_error = final_output.detach() - target_one_hot
                
                # 最終層の誤差を model_state に保存
                final_layer_name: str = self.layers[-1].name
                model_state[f'prediction_error_{final_layer_name}'] = top_down_error
                
            except Exception as e:
                if logger:
                    logger.warning(
                        f"Failed to calculate target error for P1-1: {e}. "
                        f"Output shape: {final_output.shape}, Target shape: {targets.shape}"
                    )
                pass

        # 2. 誤差をトップダウンに逆伝播（ローカル版）
        # (重みの転置 W.T を使って誤差を前の層に伝える)
        if top_down_error is not None:
            # (reversed でレイヤーを逆順に)
            for layer in reversed(self.layers):
                
                # (P4-4 の LIFLayer でないと W が取得できない)
                if not isinstance(layer, LIFLayer) or top_down_error is None:
                    continue
                    
                layer_name: str = layer.name
                
                # (P1-1) このレイヤーの出力誤差 (post_error) は
                #        model_state から取得 (最終層以外は前のループで計算済み)
                post_error: Optional[Tensor] = model_state.get(
                    f'prediction_error_{layer_name}'
                )
                
                if post_error is None:
                    # (もし何らかの理由で誤差がなければ、この層の伝播はスキップ)
                    continue

                try:
                    # 誤差の逆伝播 (ローカル版):
                    # pre_error = post_error @ W
                    # (LIFLayer の W は [N_out, N_in])
                    # (post_error は [B, N_out])
                    # (pre_error は [B, N_in])
                    
                    # (P1-2) プリシナプス活性 (順伝播時に保存済み)
                    pre_activity: Optional[Tensor] = model_state.get(
                        f'pre_activity_{layer_name}'
                    )
                    
                    if pre_activity is None:
                        continue

                    # (P1-1) プリシナプス（前段）レイヤーの誤差を計算
                    # (活性化関数の導関数 (f') は無視し、単純な線形逆伝播を行う)
                    # pre_error = post_error @ layer.W
                    pre_error: Tensor = torch.matmul(post_error, layer.W)
                    
                    # 前段のレイヤー名を見つける (ダミー: 1つ前のレイヤー)
                    current_idx: int = self.layers.index(layer)
                    if current_idx > 0:
                        pre_layer_name: str = self.layers[current_idx - 1].name
                        # (P1-1) 前段のレイヤーの出力誤差として保存
                        model_state[f'prediction_error_{pre_layer_name}'] = pre_error
                        
                except Exception as e:
                    if logger:
                        logger.error(
                            f"Failed P1-1 backward error propagation at {layer.name}: {e}"
                        )
                    pass # エラーが発生しても続行

        # --- ▲▲▲ P1-1 改善 (ダミー実装の解消) ▲▲▲ ---
        
        return model_state  # type: ignore[return-value]
