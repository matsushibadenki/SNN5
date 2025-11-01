# ファイルパス: snn_research/core/learning_rules/predictive_coding_rule.py
# タイトル: 予測符号化 (Predictive Coding) 学習規則 (P1-1 統合)
# 機能説明: 
#   Project SNN4のロードマップ (P1-1, P1-2) に基づく、
#   予測符号化（PC）の原理に従う具象学習規則。
#
#   (ダミー実装の解消 と P4-4 との連携):
#   - 'step' メソッドが、'self.layer_name' を使用して
#     'prediction_error_{self.layer_name}' (ポストシナプス誤差) と
#     'pre_activity_{self.layer_name}' (プリシナプス活性) を
#     'model_state' から正確に取得します。
#   - 'self.params' (Iterable[nn.Parameter]) から W と b を取得し、
#     PyTorch テンソル演算で更新します。

from typing import Dict, Any, Iterable, Optional, cast, List
import logging

# --- PyTorch のインポート ---
import torch
import torch.nn as nn
from torch import Tensor

# 親ディレクトリのABCをインポート
try:
    from ..learning_rule import AbstractLearningRule, Parameters
except ImportError:
    # (mypy フォールバック)
    Parameters = Iterable[nn.Parameter] # type: ignore[misc]
    from abc import ABC, abstractmethod
    class AbstractLearningRule(ABC): # type: ignore[no-redef]
        def __init__(self, params: Parameters, **kwargs: Any) -> None:
            self.params: Parameters = params
            self.hparams: Dict[str, Any] = {}
            self.layer_name: Optional[str] = kwargs.get('layer_name')
        @abstractmethod
        def step(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
            raise NotImplementedError
        @abstractmethod
        def zero_grad(self) -> None:
            raise NotImplementedError


# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)

class PredictiveCodingRule(AbstractLearningRule):
    """
    P1-1 (予測符号化) および P1-2 (ローカル学習) に基づく学習規則。
    """

    def __init__(self, params: Parameters, **kwargs: Any) -> None:
        # (self.layer_name は super() で設定される)
        super().__init__(params, **kwargs)
        
        self.error_weight: float = float(kwargs.get('error_weight', 1.0))
        self.hparams['error_weight'] = self.error_weight
        
        # --- ダミー実装の解消 (self.named_params 削除) ---
        # self.params は [W, b] のリスト (Iterable) であると仮定
        param_list: List[nn.Parameter] = list(self.params)
        if len(param_list) < 2:
            logger.warning(f"Layer {self.layer_name} passed < 2 params to LearningRule.")
        
        # (型チェックのため、W と b を明示的に保持するが、
        #  self.params がマスターリストである)
        self.W: nn.Parameter = param_list[0]
        self.b: nn.Parameter = param_list[1]


    def step(
        self,
        inputs: Tensor,
        targets: Optional[Tensor],
        model_state: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        予測符号化に基づき、単一の重み更新ステップを実行します。
        
        (ダミー実装の解消):
        self.layer_name を使用して誤差と活性を取得します。
        """
        
        lr: float = float(self.hparams.get('learning_rate', 0.01))
        
        # --- ダミー実装の解消 (self.layer_name を使用) ---
        if self.layer_name is None:
            logger.error("LearningRule missing 'layer_name'. Skipping update.")
            return {'update_magnitude': torch.tensor(0.0)}

        # 1. プリシナプス活性 (Pre-Activity) を取得
        # (P4-3 が 'pre_activity_{self.layer_name}' で保存)
        pre_activity: Optional[Tensor] = model_state.get(
            f'pre_activity_{self.layer_name}'
        )
        
        # 2. ポストシナプス誤差 (Post-Error) を取得
        # (P4-3 が 'prediction_error_{self.layer_name}' で保存)
        post_error: Optional[Tensor] = model_state.get(
            f'prediction_error_{self.layer_name}'
        )
        
        if pre_activity is None:
            logger.warning(f"'{self.layer_name}' missing 'pre_activity' in model_state.")
            return {'update_magnitude': torch.tensor(0.0)}
        if post_error is None:
            logger.warning(f"'{self.layer_name}' missing 'prediction_error' in model_state.")
            return {'update_magnitude': torch.tensor(0.0)}
        
        # (P4-3 の時間ループ全体で平均された誤差や活性を使うべきだが、
        #  ここでは簡略化のため、最終ステップの状態のみを使うと仮定)
        
        total_update_magnitude: Tensor = torch.tensor(0.0, device=inputs.device)

        # P1-2: ローカルな信用割当（BP不使用）
        with torch.no_grad():
            try:
                # 3. ヘブ則に基づく更新: delta_W = lr * post_error * pre_activity.T
                
                # (PyTorch 演算: [B, N_pre] と [B, N_post])
                # (バッチ平均を計算)
                if pre_activity.dim() == 1: pre_activity = pre_activity.unsqueeze(0)
                if post_error.dim() == 1: post_error = post_error.unsqueeze(0)

                # (B, N_post) -> (B, N_post, 1)
                error_t: Tensor = post_error.unsqueeze(2)
                # (B, N_pre) -> (B, 1, N_pre)
                pre_act_t: Tensor = pre_activity.unsqueeze(1)
                
                # (B, N_post, 1) @ (B, 1, N_pre) -> (B, N_post, N_pre)
                delta_W: Tensor = torch.bmm(error_t, pre_act_t)
                delta_W = delta_W.mean(dim=0) # バッチ平均
                
                delta_W = delta_W * (lr * self.error_weight)
                
                # バイアスの更新 (単純な誤差の平均)
                delta_b: Tensor = post_error.mean(dim=0)
                delta_b = delta_b * (lr * self.error_weight)

                # 重み (W) と バイアス (b) を更新
                if self.W.shape == delta_W.shape:
                    self.W.sub_(delta_W)
                    total_update_magnitude += delta_W.abs().sum()
                else:
                    logger.warning(f"Shape mismatch for W in {self.layer_name}")

                if self.b.shape == delta_b.shape:
                    self.b.sub_(delta_b)
                    total_update_magnitude += delta_b.abs().sum()
                else:
                    logger.warning(f"Shape mismatch for b in {self.layer_name}")
                
            except Exception as e:
                if logger:
                    logger.error(
                        f"Failed to calculate update for {self.layer_name}: {e}"
                    )
                pass

        return {'total_update_magnitude': total_update_magnitude}

    def zero_grad(self) -> None:
        pass
